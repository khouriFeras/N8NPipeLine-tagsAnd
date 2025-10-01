#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
n8n-ready API server for the automated product tagging system.

This Flask API provides REST endpoints that n8n can easily integrate with
to automate the product tagging workflow.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd

# Import our tagging modules
from gen_descriptors import main as gen_descriptors_main, parse_args as gen_descriptors_parse
from KNN_retriver import main as knn_main, parse_args as knn_parse
from bootstrap_pseudolabels import main as bootstrap_main, parse_args as bootstrap_parse
from unified_tagging_pipeline import run_full_pipeline, step1_add_tags_to_pseudo, step2_merge_with_data, step3_format_for_upload

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for n8n integration

# Global variables for caching
_taxonomy_ready = False
_taxonomy_path = "texo/taxonomy_descriptors_texo.xlsx"
_embeddings_path = "texo/node_embeddings_texo.parquet"

def check_taxonomy_ready():
    """Check if taxonomy is ready for processing."""
    global _taxonomy_ready
    if not _taxonomy_ready:
        _taxonomy_ready = (
            Path(_taxonomy_path).exists() and 
            Path(_embeddings_path).exists()
        )
    return _taxonomy_ready

def ensure_taxonomy_ready():
    """Ensure taxonomy is ready, raise error if not."""
    if not check_taxonomy_ready():
        raise ValueError(
            "Taxonomy not ready. Please run setup first: "
            "POST /api/setup-taxonomy"
        )

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "taxonomy_ready": check_taxonomy_ready()
    })

@app.route('/api/setup-taxonomy', methods=['POST'])
def setup_taxonomy():
    """Setup taxonomy descriptors and embeddings (one-time setup)."""
    try:
        data = request.get_json() or {}
        taxonomy_file = data.get('taxonomy_file', 'texo/FULL TEXO.xlsx')
        
        logger.info(f"Setting up taxonomy from: {taxonomy_file}")
        
        # Step 1: Generate descriptors
        logger.info("Step 1: Generating descriptors...")
        sys.argv = [
            'gen_descriptors.py',
            '--inp', taxonomy_file,
            '--out', _taxonomy_path,
            '--model', 'gpt-4o-mini',
            '--batch', '12',
            '--log-level', 'INFO'
        ]
        gen_descriptors_main()
        
        # Step 2: Create embeddings
        logger.info("Step 2: Creating embeddings...")
        sys.argv = [
            'KNN_retriver.py',
            '--inp', _taxonomy_path,
            '--out', _embeddings_path,
            '--model', 'text-embedding-3-large',
            '--batch', '64',
            '--faiss',
            '--log-level', 'INFO'
        ]
        knn_main()
        
        global _taxonomy_ready
        _taxonomy_ready = True
        
        return jsonify({
            "status": "success",
            "message": "Taxonomy setup completed successfully",
            "descriptors_file": _taxonomy_path,
            "embeddings_file": _embeddings_path
        })
        
    except Exception as e:
        logger.error(f"Taxonomy setup failed: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/tag-products', methods=['POST'])
def tag_products():
    """Tag products using the complete pipeline."""
    try:
        ensure_taxonomy_ready()
        
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data provided"}), 400
        
        # Extract parameters
        products_file = data.get('products_file')
        if not products_file:
            return jsonify({"status": "error", "message": "products_file is required"}), 400
        
        output_dir = data.get('output_dir', 'output')
        pseudo_id_col = data.get('pseudo_id_col', 'product_id')
        data_id_col = data.get('data_id_col', 'Title')
        
        logger.info(f"Tagging products from: {products_file}")
        
        # Run the complete pipeline
        result = run_full_pipeline(
            pseudo_file="temp_pseudo_labels.csv",  # Will be generated
            descriptors_file=_taxonomy_path,
            data_file=products_file,
            output_dir=output_dir,
            pseudo_id_col=pseudo_id_col,
            data_id_col=data_id_col,
            log_level="INFO"
        )
        
        # Generate pseudo labels first
        logger.info("Generating pseudo labels...")
        sys.argv = [
            'bootstrap_pseudolabels.py',
            '--nodes', _embeddings_path,
            '--descriptors', _taxonomy_path,
            '--products', products_file,
            '--title-col', data_id_col,
            '--desc-col', 'Body (HTML)',
            '--embed-model', 'text-embedding-3-large',
            '--llm-model', 'gpt-4o-mini',
            '--topk', '8',
            '--conf-thresh', '0.85',
            '--out', 'temp_pseudo_labels.csv',
            '--log-level', 'INFO'
        ]
        bootstrap_main()
        
        # Now run the full pipeline
        result = run_full_pipeline(
            pseudo_file="temp_pseudo_labels.csv",
            descriptors_file=_taxonomy_path,
            data_file=products_file,
            output_dir=output_dir,
            pseudo_id_col=pseudo_id_col,
            data_id_col=data_id_col,
            log_level="INFO"
        )
        
        # Clean up temp file
        Path("temp_pseudo_labels.csv").unlink(missing_ok=True)
        
        return jsonify({
            "status": "success",
            "message": "Products tagged successfully",
            "output_files": {
                "pseudo_labels": f"{output_dir}/pseudo_labels_with_tags.csv",
                "merged_data": f"{output_dir}/merged_data_with_tags.xlsx",
                "upload_template": f"{output_dir}/upload_template_final.xlsx"
            },
            "stats": {
                "total_products": len(result["upload_template"]),
                "tagged_products": (result["upload_template"]["tags"] != "").sum()
            }
        })
        
    except Exception as e:
        logger.error(f"Product tagging failed: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/tag-products-simple', methods=['POST'])
def tag_products_simple():
    """Simplified tagging endpoint that returns just the tagged data."""
    try:
        ensure_taxonomy_ready()
        
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No JSON data provided"}), 400
        
        # Extract parameters
        products_data = data.get('products_data')  # Array of product objects
        if not products_data:
            return jsonify({"status": "error", "message": "products_data is required"}), 400
        
        # Convert products data to DataFrame and save as temp file
        df = pd.DataFrame(products_data)
        temp_file = f"temp_products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(temp_file, index=False)
        
        try:
            # Generate pseudo labels
            sys.argv = [
                'bootstrap_pseudolabels.py',
                '--nodes', _embeddings_path,
                '--descriptors', _taxonomy_path,
                '--products', temp_file,
                '--title-col', 'Title',
                '--desc-col', 'Body (HTML)',
                '--embed-model', 'text-embedding-3-large',
                '--llm-model', 'gpt-4o-mini',
                '--topk', '8',
                '--conf-thresh', '0.85',
                '--out', 'temp_pseudo.csv',
                '--log-level', 'INFO'
            ]
            bootstrap_main()
            
            # Add tags to pseudo labels
            pseudo_with_tags = step1_add_tags_to_pseudo(
                pseudo_file="temp_pseudo.csv",
                descriptors_file=_taxonomy_path,
                output_file="temp_pseudo_with_tags.csv"
            )
            
            # Merge with original data
            merged_data = step2_merge_with_data(
                pseudo_file="temp_pseudo_with_tags.csv",
                data_file=temp_file,
                output_file="temp_merged.xlsx",
                pseudo_id_col="product_id",
                data_id_col="Title"
            )
            
            # Format for upload
            upload_template = step3_format_for_upload(
                pseudo_file="temp_pseudo_with_tags.csv",
                products_file=temp_file,
                output_file="temp_upload.xlsx",
                pseudo_id_col="product_id",
                prod_id_col="Title"
            )
            
            # Convert result to JSON-serializable format
            result_data = upload_template.to_dict('records')
            
            return jsonify({
                "status": "success",
                "message": "Products tagged successfully",
                "data": result_data,
                "stats": {
                    "total_products": len(result_data),
                    "tagged_products": sum(1 for item in result_data if item.get('tags', '').strip())
                }
            })
            
        finally:
            # Clean up temp files
            for temp_file_path in [temp_file, "temp_pseudo.csv", "temp_pseudo_with_tags.csv", "temp_merged.xlsx", "temp_upload.xlsx"]:
                Path(temp_file_path).unlink(missing_ok=True)
        
    except Exception as e:
        logger.error(f"Simple product tagging failed: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """Download generated files."""
    try:
        file_path = Path(filename)
        if not file_path.exists():
            return jsonify({"status": "error", "message": "File not found"}), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        logger.error(f"File download failed: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status and available endpoints."""
    return jsonify({
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "taxonomy_ready": check_taxonomy_ready(),
        "endpoints": {
            "health": "GET /api/health",
            "setup": "POST /api/setup-taxonomy",
            "tag_products": "POST /api/tag-products",
            "tag_products_simple": "POST /api/tag-products-simple",
            "download": "GET /api/download/<filename>",
            "status": "GET /api/status"
        },
        "required_env": ["OPENAI_API_KEY"]
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"status": "error", "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"status": "error", "message": "Internal server error"}), 500

if __name__ == '__main__':
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        logger.warning("OPENAI_API_KEY not set. Some endpoints will not work.")
    
    # Create necessary directories
    Path("texo").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    
    # Get port from environment (Railway sets this automatically)
    port = int(os.environ.get('PORT', 5000))
    
    logger.info("Starting n8n-ready API server...")
    logger.info(f"Running on port {port}")
    logger.info("Available endpoints:")
    logger.info("  GET  /api/health - Health check")
    logger.info("  POST /api/setup-taxonomy - Setup taxonomy (one-time)")
    logger.info("  POST /api/tag-products - Tag products (file-based)")
    logger.info("  POST /api/tag-products-simple - Tag products (JSON-based)")
    logger.info("  GET  /api/download/<filename> - Download files")
    logger.info("  GET  /api/status - System status")
    
    app.run(host='0.0.0.0', port=port, debug=False)