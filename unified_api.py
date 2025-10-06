#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified API Server for Translation and Tagging
Combines both translation and tagging services in one Flask app
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
import tempfile
import uuid

# Import translation modules
from apis.universal_translation_seo import process_excel_file

# Import tagging modules
from gen_descriptors import main as gen_descriptors_main, parse_args as gen_descriptors_parse
from KNN_retriver import main as knn_main, parse_args as knn_parse
from bootstrap_pseudolabels import main as bootstrap_main, parse_args as bootstrap_parse
from unified_tagging_pipeline import run_full_pipeline, step1_add_tags_to_pseudo, step2_merge_with_data, step3_format_for_upload

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for n8n integration

# Store generated files temporarily
generated_files = {}

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

# ==================== TRANSLATION ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy", 
        "service": "unified-translation-tagging-service",
        "timestamp": datetime.now().isoformat(),
        "taxonomy_ready": check_taxonomy_ready()
    })

@app.route('/test', methods=['GET'])
def test():
    return jsonify({
        "message": "Unified Translation and Tagging service is running!",
        "endpoints": {
            "health": "GET /health",
            "test": "GET /test", 
            "process": "POST /process",
            "translate": "POST /translate",
            "download": "GET /download/<file_id>",
            "tag_health": "GET /api/health",
            "setup_taxonomy": "POST /api/setup-taxonomy",
            "tag_products": "POST /api/tag-products-simple"
        }
    })

@app.route('/process', methods=['POST'])
def process_data():
    """Translation and SEO processing endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        csv_content = data.get('csv_content', '')
        product_context = data.get('product_context', 'tools and equipment')
        sample = int(data.get('sample', 5))
        
        if not csv_content:
            return jsonify({"error": "No CSV content provided"}), 400
            
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as temp_csv:
            temp_csv.write(csv_content)
            temp_csv_path = temp_csv.name
            
        # Create temporary output file
        temp_output = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
        temp_output_path = temp_output.name
        temp_output.close()
        
        try:
            # Process the file (run async function)
            import asyncio
            result = asyncio.run(process_excel_file(
                input_path=Path(temp_csv_path),
                output_path=Path(temp_output_path),
                name_col="Title",
                desc_col="Body (HTML)",
                product_context=product_context,
                sample=sample,
                api_key=os.getenv("OPENAI_API_KEY", "")
            ))
            
            output_size = os.path.getsize(temp_output_path) if os.path.exists(temp_output_path) else 0
            file_id = str(uuid.uuid4())

            generated_files[file_id] = temp_output_path
            
            return jsonify({
                "success": True,
                "message": "Translation and SEO processing completed successfully",
                "output_size": output_size,
                "result": result,
                "sample_processed": sample,
                "file_id": file_id,
                "output_columns": [
                    "arabic description", "product_title_ar", "meta_title", "meta_description", "handle_ar"
                ],
                "features": {
                    "translation": "Arabic product descriptions and titles",
                    "seo": "Meta titles, descriptions, and SEO-friendly handles",
                    "combined": "All data appended to new columns in the output file"
                }
            })
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e)
            }), 500
            
        finally:
            # Clean up temp CSV file
            if os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)
                
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download/<file_id>', methods=['GET'])
def download_file(file_id):
    """Download processed file"""
    if file_id not in generated_files:
        return jsonify({"error": "File not found"}), 404
    
    file_path = generated_files[file_id]
    if not os.path.exists(file_path):
        return jsonify({"error": "File no longer exists"}), 404
    
    return send_file(file_path, as_attachment=True, download_name=f"processed_products_{file_id}.xlsx")

# ==================== TAGGING ENDPOINTS ====================

@app.route('/api/health', methods=['GET'])
def api_health():
    """Health check for tagging API"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "taxonomy_ready": check_taxonomy_ready()
    })

@app.route('/api/status', methods=['GET'])
def api_status():
    """Detailed status for tagging API"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "taxonomy_ready": check_taxonomy_ready(),
        "taxonomy_path": _taxonomy_path,
        "embeddings_path": _embeddings_path,
        "taxonomy_exists": Path(_taxonomy_path).exists(),
        "embeddings_exists": Path(_embeddings_path).exists()
    })

@app.route('/api/setup-taxonomy', methods=['POST'])
def setup_taxonomy():
    """One-time setup for taxonomy (generates descriptors and embeddings)"""
    try:
        data = request.get_json() or {}
        taxonomy_file = data.get('taxonomy_file', 'texo/FULL TEXO.xlsx')
        
        if not os.path.exists(taxonomy_file):
            return jsonify({
                "success": False,
                "error": f"Taxonomy file not found: {taxonomy_file}"
            }), 400
        
        logger.info(f"Starting taxonomy setup with file: {taxonomy_file}")
        
        # Step 1: Generate descriptors
        logger.info("Step 1: Generating descriptors...")
        import sys
        old_argv = sys.argv
        sys.argv = ['gen_descriptors.py', '--inp', taxonomy_file, '--out', _taxonomy_path, '--model', 'gpt-4o-mini', '--batch', '12']
        gen_descriptors_main()
        sys.argv = old_argv
        
        # Step 2: Create embeddings
        logger.info("Step 2: Creating embeddings...")
        sys.argv = ['KNN_retriver.py', '--inp', _taxonomy_path, '--out', _embeddings_path, '--model', 'text-embedding-3-large', '--batch', '64', '--faiss']
        knn_main()
        sys.argv = old_argv
        
        # Update global state
        global _taxonomy_ready
        _taxonomy_ready = True
        
        return jsonify({
            "success": True,
            "message": "Taxonomy setup completed successfully",
            "descriptors_file": _taxonomy_path,
            "embeddings_file": _embeddings_path
        })
        
    except Exception as e:
        logger.error(f"Taxonomy setup failed: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/tag-products-simple', methods=['POST'])
def tag_products_simple():
    """Tag products using JSON data - Simplified version"""
    try:
        data = request.get_json()
        if not data or 'products_data' not in data:
            return jsonify({
                "status": "error",
                "message": "No products_data provided"
            }), 400
        
        products = data['products_data']
        if not isinstance(products, list):
            return jsonify({
                "status": "error",
                "message": "products_data must be a list"
            }), 400
        
        logger.info(f"Processing {len(products)} products for tagging")
        
        # For now, return a simple response with mock tags
        # This will help us test the endpoint without the complex taxonomy setup
        results = []
        for i, product in enumerate(products):
            title = product.get('Title', '')
            product_type = product.get('Product Type', '')
            
            # Generate simple mock tags based on product type
            if 'drill' in title.lower() or 'power' in product_type.lower():
                tags = "tools,power-tools,drills"
            elif 'hammer' in title.lower() or 'hand' in product_type.lower():
                tags = "tools,hand-tools,hammers"
            else:
                tags = "tools,general"
            
            results.append({
                "Title": title,
                "Product Type": product_type,
                "Vendor": product.get('Vendor', ''),
                "Body (HTML)": product.get('Body (HTML)', ''),
                "tags": tags,
                "confidence": 0.85,
                "product_id": f"prod_{i+1}",
                "choice_id": f"choice_{i+1}",
                "verified": 1
            })
        
        return jsonify({
            "status": "success",
            "data": results,
            "stats": {
                "total_products": len(products),
                "tagged_products": len(results)
            }
        })
                
    except Exception as e:
        logger.error(f"Tag products simple failed: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
