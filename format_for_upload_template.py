#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Format pseudo labels output to match Upload Product Template .xlsx structure.

This script takes the pseudo_labels.csv output (with tags) and formats it
to match the expected upload template format with Tags as the first column.
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
    return logging.getLogger("format_template")

def safe_str(v) -> str:
    """Safely convert value to string, handling None and NaN."""
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"", "nan", "none", "null", "n/a", "-"} else s

def format_for_upload_template(pseudo_file: str, products_file: str, output_file: str, 
                              pseudo_id_col: str = "product_id", 
                              prod_id_col: str = "Title",
                              log_level: str = "INFO"):
    """
    Format pseudo labels to match Upload Product Template structure.
    
    Args:
        pseudo_file: Path to pseudo_labels.csv (with tags)
        products_file: Path to products Excel file
        output_file: Path to output Excel file
        pseudo_id_col: Column name in pseudo file for joining
        prod_id_col: Column name in products file for joining
        log_level: Logging level
    """
    logger = setup_logger(log_level)
    
    # Load pseudo labels
    logger.info(f"Loading pseudo labels from: {pseudo_file}")
    pseudo_df = pd.read_csv(pseudo_file, encoding="utf-8")
    
    # Load products
    logger.info(f"Loading products from: {products_file}")
    prod_path = Path(products_file)
    if prod_path.suffix.lower() in [".xlsx", ".xls"]:
        products_df = pd.read_excel(prod_path, engine="openpyxl")
    elif prod_path.suffix.lower() == ".csv":
        products_df = pd.read_csv(prod_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported products file type: {prod_path.suffix}")
    
    # Check required columns
    if "tags" not in pseudo_df.columns:
        raise ValueError("Pseudo labels file must contain 'tags' column")
    
    if pseudo_id_col not in pseudo_df.columns:
        raise ValueError(f"Pseudo labels missing column: {pseudo_id_col}")
    
    if prod_id_col not in products_df.columns:
        raise ValueError(f"Products missing column: {prod_id_col}")
    
    logger.info(f"Pseudo labels: {len(pseudo_df)} rows")
    logger.info(f"Products: {len(products_df)} rows")
    
    # Prepare join keys
    pseudo_df["_join_key"] = pseudo_df[pseudo_id_col].astype(str).str.strip()
    products_df["_join_key"] = products_df[prod_id_col].astype(str).str.strip()
    
    # Merge pseudo labels with products
    logger.info("Merging pseudo labels with products...")
    merged_df = products_df.merge(
        pseudo_df[["_join_key", "tags", "choice_id", "level", "confidence", "verified"]], 
        on="_join_key", 
        how="left"
    )
    
    # Clean up join key
    merged_df = merged_df.drop(columns=["_join_key"])
    
    # Create the template structure
    logger.info("Creating upload template format...")
    
    # Start with the template structure
    template_df = pd.DataFrame()
    
    # Tags column (first column as per template)
    template_df["Tags"] = merged_df["tags"].fillna("")
    
    # Product Type (can be derived from tags or set default)
    template_df["Product Type "] = "Pet Supplies"  # Default based on your data
    
    # Title from products (prefer product_title_ar or other title-like columns)
    title_candidates = [
        "product_title_ar",
        "Title",
        "اسم المادة",
        "name",
        "Name",
    ]
    title_col = next((c for c in title_candidates if c in merged_df.columns), None)
    if title_col:
        template_df["Title"] = merged_df[title_col].fillna("")
    else:
        # Fallback to the join id column only if no better title is present
        template_df["Title"] = merged_df[prod_id_col].fillna("")
    
    # Variant SKU - try to find SKU column or use product ID
    sku_cols = ["رقم المادة", "رقم الباركود", "Variant SKU", "SKU", "Product ID"]
    sku_col = None
    for col in sku_cols:
        if col in merged_df.columns:
            sku_col = col
            break
    
    if sku_col:
        template_df["Variant SKU"] = merged_df[sku_col].fillna("")
    else:
        template_df["Variant SKU"] = merged_df[prod_id_col].fillna("")
    
    # Variant Price - try to find price column
    price_cols = ["سعر المفرق", "Price", "Variant Price", "Retail Price"]
    price_col = None
    for col in price_cols:
        if col in merged_df.columns:
            price_col = col
            break
    
    if price_col:
        template_df["Variant Price"] = merged_df[price_col].fillna("")
    else:
        template_df["Variant Price"] = ""
    
    # Variant Cost - try to find cost column
    cost_cols = ["سعر الجملة", "Cost", "Variant Cost", "Wholesale Price"]
    cost_col = None
    for col in cost_cols:
        if col in merged_df.columns:
            cost_col = col
            break
    
    if cost_col:
        template_df["Variant Cost"] = merged_df[cost_col].fillna("")
    else:
        template_df["Variant Cost"] = ""
    
    # Body (HTML) - prefer Arabic description when available
    desc_cols = ["arabic description", "Description", "Body (HTML)", "meta_description"]
    desc_col = None
    for col in desc_cols:
        if col in merged_df.columns:
            desc_col = col
            break
    
    if desc_col:
        template_df["Body (HTML)"] = merged_df[desc_col].fillna("")
    else:
        template_df["Body (HTML)"] = ""
    
    # Vendor - try to find brand/vendor column
    vendor_cols = ["Brand", "Vendor", "Manufacturer"]
    vendor_col = None
    for col in vendor_cols:
        if col in merged_df.columns:
            vendor_col = col
            break
    
    if vendor_col:
        template_df["Vendor"] = merged_df[vendor_col].fillna("")
    else:
        template_df["Vendor"] = ""
    
    # Image Src - try to find image URL column
    img_cols = ["IMGURL1", "Image Src", "Image URL", "Main Image"]
    img_col = None
    for col in img_cols:
        if col in merged_df.columns:
            img_col = col
            break
    
    if img_col:
        template_df["Image Src"] = merged_df[img_col].fillna("")
    else:
        template_df["Image Src"] = ""
    
    # Status
    template_df["Status"] = "active"
    
    # Published
    template_df["Published "] = True
    
    # Total Inventory Qty
    template_df["Total Inventory Qty"] = 0
    
    # Metafield: custom.brand
    if vendor_col:
        template_df["Metafield: custom.brand"] = merged_df[vendor_col].fillna("")
    else:
        template_df["Metafield: custom.brand"] = ""
    
    # Add metadata columns for reference (hidden or can be removed)
    template_df["_choice_id"] = merged_df["choice_id"].fillna("")
    template_df["_level"] = merged_df["level"].fillna("")
    template_df["_confidence"] = merged_df["confidence"].fillna(0.0)
    template_df["_verified"] = merged_df["verified"].fillna(0)
    
    # Clean up empty tags
    template_df["Tags"] = template_df["Tags"].apply(safe_str)
    
    # Remove rows with no tags (optional)
    # template_df = template_df[template_df["Tags"].str.len() > 0]
    
    logger.info(f"Formatted data: {len(template_df)} rows")
    logger.info(f"Rows with tags: {len(template_df[template_df['Tags'].str.len() > 0])}")
    
    # Save to Excel
    logger.info(f"Saving to: {output_file}")
    template_df.to_excel(output_file, index=False, engine="openpyxl")
    
    logger.info("✅ Upload template formatted successfully!")
    
    # Print summary
    print("\n=== FORMATTED UPLOAD TEMPLATE SUMMARY ===")
    print(f"Output file: {output_file}")
    print(f"Total rows: {len(template_df)}")
    print(f"Rows with tags: {len(template_df[template_df['Tags'].str.len() > 0])}")
    print(f"Columns: {list(template_df.columns)}")
    
    # Show sample tags
    tagged_rows = template_df[template_df["Tags"].str.len() > 0]
    if len(tagged_rows) > 0:
        print("\nSample tags:")
        for i, (_, row) in enumerate(tagged_rows.head(3).iterrows()):
            title_str = str(row.get('Title', ''))
            tags_str = str(row.get('Tags', ''))
            print(f"  {i+1}. {title_str[:50]}... -> {tags_str}")
    
    return template_df

def main():
    parser = argparse.ArgumentParser(description="Format pseudo labels to Upload Product Template format")
    parser.add_argument("--pseudo", required=True, help="pseudo_labels.csv file (with tags)")
    parser.add_argument("--products", required=True, help="Products Excel/CSV file")
    parser.add_argument("--output", default="upload_template_formatted.xlsx", help="Output Excel file")
    parser.add_argument("--pseudo-id-col", default="product_id", help="ID column in pseudo file")
    parser.add_argument("--prod-id-col", default="Title", help="ID column in products file")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    
    args = parser.parse_args()
    
    try:
        format_for_upload_template(
            pseudo_file=args.pseudo,
            products_file=args.products,
            output_file=args.output,
            pseudo_id_col=args.pseudo_id_col,
            prod_id_col=args.prod_id_col,
            log_level=args.log_level
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
