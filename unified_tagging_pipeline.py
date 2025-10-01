#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Tagging Pipeline - Combines three tagging operations into one script:

1. add_tags_to_existing_pseudo.py - Add hierarchical tags to pseudo labels
2. append_tags_to_DATA.py - Merge pseudo labels with main data file  
3. format_for_upload_template.py - Format for upload template

This script provides a complete pipeline from pseudo labels to final upload format.
"""

import argparse
import pandas as pd
import sys
from pathlib import Path
import logging

def setup_logger(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger("unified_pipeline")

def safe_str(v) -> str:
    """Safely convert value to string, handling None and NaN."""
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"", "nan", "none", "null", "n/a", "-"} else s

def build_hierarchical_tags(row) -> str:
    """Build hierarchical tags from L1_en, L2_en, L3_en, L4_en columns."""
    tags = []
    for level in ["L1_en", "L2_en", "L3_en", "L4_en"]:
        if level in row and pd.notna(row[level]) and str(row[level]).strip():
            tag = str(row[level]).strip()
            if tag.lower() not in {"", "nan", "none", "null", "n/a", "-"}:
                tags.append(tag)
    return ",".join(tags)

def step1_add_tags_to_pseudo(pseudo_file: str, descriptors_file: str, output_file: str, 
                            descriptors_sheet: str = "descriptors", logger=None):
    """
    Step 1: Add tags column to existing pseudo labels file.
    
    Args:
        pseudo_file: Path to existing pseudo_labels.csv
        descriptors_file: Path to taxonomy_descriptors.xlsx
        output_file: Path to output CSV file with tags
        descriptors_sheet: Sheet name in descriptors file
        logger: Logger instance
    """
    if logger is None:
        logger = setup_logger()
    
    logger.info("Step 1: Adding hierarchical tags to pseudo labels...")
    
    # Load pseudo labels
    logger.info(f"Loading pseudo labels from: {pseudo_file}")
    pseudo_df = pd.read_csv(pseudo_file)
    logger.info(f"Loaded {len(pseudo_df)} pseudo labels")
    
    # Load descriptors
    logger.info(f"Loading descriptors from: {descriptors_file}")
    descriptors_df = pd.read_excel(descriptors_file, sheet_name=descriptors_sheet)
    logger.info(f"Loaded {len(descriptors_df)} descriptors")
    
    # Create lookup dictionary for tags
    logger.info("Building tag lookup dictionary...")
    tag_lookup = {}
    for _, row in descriptors_df.iterrows():
        node_id = safe_str(row.get("node_id", ""))
        if node_id:
            tags = build_hierarchical_tags(row)
            tag_lookup[node_id] = tags
    
    logger.info(f"Created lookup for {len(tag_lookup)} nodes")
    
    # Add tags column to pseudo labels
    logger.info("Adding tags column...")
    pseudo_df["tags"] = pseudo_df["choice_id"].map(tag_lookup).fillna("")
    
    # Count how many got tags
    tagged_count = (pseudo_df["tags"] != "").sum()
    logger.info(f"Added tags to {tagged_count}/{len(pseudo_df)} pseudo labels")
    
    # Save result
    pseudo_df.to_csv(output_file, index=False)
    logger.info(f"Saved pseudo labels with tags to: {output_file}")
    
    return pseudo_df

def step2_merge_with_data(pseudo_file: str, data_file: str, output_file: str,
                         pseudo_id_col: str = "product_id", 
                         data_id_col: str = "Title",
                         logger=None):
    """
    Step 2: Merge pseudo labels with main data file.
    
    Args:
        pseudo_file: Path to pseudo_labels.csv (with tags)
        data_file: Path to main data Excel/CSV file
        output_file: Path to output merged file
        pseudo_id_col: Column name in pseudo file for joining
        data_id_col: Column name in data file for joining
        logger: Logger instance
    """
    if logger is None:
        logger = setup_logger()
    
    logger.info("Step 2: Merging pseudo labels with main data...")
    
    # Load pseudo labels with tags
    logger.info(f"Loading pseudo labels from: {pseudo_file}")
    pseudo_df = pd.read_csv(pseudo_file)
    logger.info(f"Loaded {len(pseudo_df)} pseudo labels")
    
    # Load main data
    logger.info(f"Loading main data from: {data_file}")
    if data_file.endswith('.xlsx') or data_file.endswith('.xls'):
        data_df = pd.read_excel(data_file)
    else:
        data_df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(data_df)} data records")
    
    # Check if ID columns exist
    if pseudo_id_col not in pseudo_df.columns:
        logger.error(f"Column '{pseudo_id_col}' not found in pseudo file. Available: {list(pseudo_df.columns)}")
        raise ValueError(f"Column '{pseudo_id_col}' not found in pseudo file")
    
    if data_id_col not in data_df.columns:
        logger.error(f"Column '{data_id_col}' not found in data file. Available: {list(data_df.columns)}")
        raise ValueError(f"Column '{data_id_col}' not found in data file")
    
    # Convert ID columns to string to avoid type mismatch
    logger.info("Converting ID columns to string for merging...")
    data_df[data_id_col] = data_df[data_id_col].astype(str)
    
    # Use title column for matching if product_id is NaN
    if pseudo_df[pseudo_id_col].isna().all():
        logger.info("product_id column is empty, using title column for matching")
        pseudo_df["title"] = pseudo_df["title"].astype(str)
        merge_key_pseudo = "title"
    else:
        pseudo_df[pseudo_id_col] = pseudo_df[pseudo_id_col].astype(str)
        merge_key_pseudo = pseudo_id_col
    
    # Merge data
    logger.info(f"Merging on pseudo.{merge_key_pseudo} = data.{data_id_col}")
    merged_df = data_df.merge(
        pseudo_df[["product_id", "title", "choice_id", "tags", "confidence", "verified"]], 
        left_on=data_id_col, 
        right_on=merge_key_pseudo, 
        how="left"
    )
    
    # Count matches
    matched_count = merged_df["choice_id"].notna().sum()
    logger.info(f"Matched {matched_count}/{len(merged_df)} records with pseudo labels")
    
    # Save result
    if output_file.endswith('.xlsx'):
        merged_df.to_excel(output_file, index=False)
    else:
        merged_df.to_csv(output_file, index=False)
    logger.info(f"Saved merged data to: {output_file}")
    
    return merged_df

def step3_format_for_upload(pseudo_file: str, products_file: str, output_file: str,
                           pseudo_id_col: str = "product_id", 
                           prod_id_col: str = "Title",
                           logger=None):
    """
    Step 3: Format for upload template with Tags as first column.
    
    Args:
        pseudo_file: Path to pseudo_labels.csv (with tags)
        products_file: Path to products Excel/CSV file
        output_file: Path to output Excel file
        pseudo_id_col: Column name in pseudo file for joining
        prod_id_col: Column name in products file for joining
        logger: Logger instance
    """
    if logger is None:
        logger = setup_logger()
    
    logger.info("Step 3: Formatting for upload template...")
    
    # Load pseudo labels with tags
    logger.info(f"Loading pseudo labels from: {pseudo_file}")
    pseudo_df = pd.read_csv(pseudo_file)
    logger.info(f"Loaded {len(pseudo_df)} pseudo labels")
    
    # Load products
    logger.info(f"Loading products from: {products_file}")
    if products_file.endswith('.xlsx') or products_file.endswith('.xls'):
        products_df = pd.read_excel(products_file)
    else:
        products_df = pd.read_csv(products_file)
    logger.info(f"Loaded {len(products_df)} products")
    
    # Convert ID columns to string to avoid type mismatch
    logger.info("Converting ID columns to string for merging...")
    products_df[prod_id_col] = products_df[prod_id_col].astype(str)
    
    # Use title column for matching if product_id is NaN
    if pseudo_df[pseudo_id_col].isna().all():
        logger.info("product_id column is empty, using title column for matching")
        pseudo_df["title"] = pseudo_df["title"].astype(str)
        merge_key_pseudo = "title"
    else:
        pseudo_df[pseudo_id_col] = pseudo_df[pseudo_id_col].astype(str)
        merge_key_pseudo = pseudo_id_col
    
    # Merge pseudo labels with products
    logger.info(f"Merging on pseudo.{merge_key_pseudo} = products.{prod_id_col}")
    merged_df = products_df.merge(
        pseudo_df[["product_id", "title", "choice_id", "tags", "confidence", "verified"]], 
        left_on=prod_id_col, 
        right_on=merge_key_pseudo, 
        how="left"
    )
    
    # Reorder columns to put Tags first
    logger.info("Reordering columns with Tags first...")
    cols = list(merged_df.columns)
    if "tags" in cols:
        # Move tags to first position
        cols.remove("tags")
        cols.insert(0, "tags")
        merged_df = merged_df[cols]
    
    # Fill missing tags with empty string
    merged_df["tags"] = merged_df["tags"].fillna("")
    
    # Count tagged products
    tagged_count = (merged_df["tags"] != "").sum()
    logger.info(f"Formatted {tagged_count}/{len(merged_df)} products with tags")
    
    # Save result
    merged_df.to_excel(output_file, index=False)
    logger.info(f"Saved upload template to: {output_file}")
    
    return merged_df

def run_full_pipeline(pseudo_file: str, descriptors_file: str, data_file: str, 
                     output_dir: str = "output",
                     descriptors_sheet: str = "descriptors",
                     pseudo_id_col: str = "product_id",
                     data_id_col: str = "Title",
                     log_level: str = "INFO"):
    """
    Run the complete unified tagging pipeline.
    
    Args:
        pseudo_file: Path to pseudo_labels.csv
        descriptors_file: Path to taxonomy_descriptors.xlsx
        data_file: Path to main data file
        output_dir: Directory for output files
        descriptors_sheet: Sheet name in descriptors file
        pseudo_id_col: Column name in pseudo file for joining
        data_id_col: Column name in data file for joining
        log_level: Logging level
    """
    logger = setup_logger(log_level)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("UNIFIED TAGGING PIPELINE STARTED")
    logger.info("=" * 60)
    
    try:
        # Step 1: Add tags to pseudo labels
        pseudo_with_tags_file = output_path / "pseudo_labels_with_tags.csv"
        pseudo_with_tags_df = step1_add_tags_to_pseudo(
            pseudo_file=pseudo_file,
            descriptors_file=descriptors_file,
            output_file=str(pseudo_with_tags_file),
            descriptors_sheet=descriptors_sheet,
            logger=logger
        )
        
        # Step 2: Merge with main data
        merged_file = output_path / "merged_data_with_tags.xlsx"
        merged_df = step2_merge_with_data(
            pseudo_file=str(pseudo_with_tags_file),
            data_file=data_file,
            output_file=str(merged_file),
            pseudo_id_col=pseudo_id_col,
            data_id_col=data_id_col,
            logger=logger
        )
        
        # Step 3: Format for upload template
        upload_template_file = output_path / "upload_template_final.xlsx"
        upload_df = step3_format_for_upload(
            pseudo_file=str(pseudo_with_tags_file),
            products_file=data_file,
            output_file=str(upload_template_file),
            pseudo_id_col=pseudo_id_col,
            prod_id_col=data_id_col,
            logger=logger
        )
        
        logger.info("=" * 60)
        logger.info("UNIFIED TAGGING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Output files created in: {output_path}")
        logger.info(f"1. Pseudo labels with tags: {pseudo_with_tags_file}")
        logger.info(f"2. Merged data: {merged_file}")
        logger.info(f"3. Upload template: {upload_template_file}")
        
        return {
            "pseudo_with_tags": pseudo_with_tags_df,
            "merged_data": merged_df,
            "upload_template": upload_df
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Unified Tagging Pipeline - Complete pipeline from pseudo labels to upload template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python unified_tagging_pipeline.py \\
    --pseudo "test_tans_pseudo_labels.csv" \\
    --descriptors "texo/taxonomy_descriptors_texo.xlsx" \\
    --data "test tans.xlsx" \\
    --output-dir "output"
  
  # Run individual steps
  python unified_tagging_pipeline.py --step add-tags \\
    --pseudo "test_tans_pseudo_labels.csv" \\
    --descriptors "texo/taxonomy_descriptors_texo.xlsx" \\
    --output "pseudo_with_tags.csv"
        """
    )
    
    # Main arguments
    parser.add_argument("--pseudo", required=True, help="Pseudo labels CSV file")
    parser.add_argument("--descriptors", required=True, help="Taxonomy descriptors Excel file")
    parser.add_argument("--data", help="Main data file (Excel/CSV)")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--output", help="Output file (for single step mode)")
    
    # Step selection
    parser.add_argument("--step", choices=["add-tags", "merge-data", "format-upload", "full"], 
                       default="full", help="Which step to run")
    
    # File options
    parser.add_argument("--descriptors-sheet", default="descriptors", help="Descriptors sheet name")
    parser.add_argument("--pseudo-id-col", default="product_id", help="ID column in pseudo file")
    parser.add_argument("--data-id-col", default="Title", help="ID column in data file")
    
    # Logging
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    
    args = parser.parse_args()
    
    try:
        if args.step == "add-tags":
            if not args.output:
                args.output = "pseudo_labels_with_tags.csv"
            step1_add_tags_to_pseudo(
                pseudo_file=args.pseudo,
                descriptors_file=args.descriptors,
                output_file=args.output,
                descriptors_sheet=args.descriptors_sheet
            )
            
        elif args.step == "merge-data":
            if not args.data:
                print("❌ Error: --data is required for merge-data step")
                sys.exit(1)
            if not args.output:
                args.output = "merged_data_with_tags.xlsx"
            step2_merge_with_data(
                pseudo_file=args.pseudo,
                data_file=args.data,
                output_file=args.output,
                pseudo_id_col=args.pseudo_id_col,
                data_id_col=args.data_id_col
            )
            
        elif args.step == "format-upload":
            if not args.data:
                print("❌ Error: --data is required for format-upload step")
                sys.exit(1)
            if not args.output:
                args.output = "upload_template_final.xlsx"
            step3_format_for_upload(
                pseudo_file=args.pseudo,
                products_file=args.data,
                output_file=args.output,
                pseudo_id_col=args.pseudo_id_col,
                prod_id_col=args.data_id_col
            )
            
        elif args.step == "full":
            if not args.data:
                print("❌ Error: --data is required for full pipeline")
                sys.exit(1)
            run_full_pipeline(
                pseudo_file=args.pseudo,
                descriptors_file=args.descriptors,
                data_file=args.data,
                output_dir=args.output_dir,
                descriptors_sheet=args.descriptors_sheet,
                pseudo_id_col=args.pseudo_id_col,
                data_id_col=args.data_id_col,
                log_level=args.log_level
            )
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
