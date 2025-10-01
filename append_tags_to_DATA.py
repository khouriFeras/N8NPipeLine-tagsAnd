#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Append hierarchical tags to DUVOtranslated.xlsx in Upload Product Template format.

This script takes the DUVOtranslated.xlsx file and pseudo_labels.csv (with tags)
and creates a new file in the Upload Product Template format with tags appended.
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
    return logging.getLogger("append_tags")

def safe_str(v) -> str:
    """Safely convert value to string, handling None and NaN."""
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"", "nan", "none", "null", "n/a", "-"} else s

def append_tags_to_duvo(duvo_file: str, pseudo_file: str, output_file: str, 
                       pseudo_id_col: str = "product_id", 
                       duvo_id_col: str = "كود المنتج - SKU",
                       log_level: str = "INFO"):
    """
    Append tags to DUVOtranslated.xlsx and format for Upload Product Template.
    
    Args:
        duvo_file: Path to DUVOtranslated.xlsx
        pseudo_file: Path to pseudo_labels.csv (with tags)
        output_file: Path to output Excel file
        pseudo_id_col: Column name in pseudo file for joining
        duvo_id_col: Column name in DUVO file for joining
        log_level: Logging level
    """
    logger = setup_logger(log_level)
    
    # Load DUVO data
    logger.info(f"Loading DUVO data from: {duvo_file}")
    duvo_df = pd.read_excel(duvo_file, engine="openpyxl")
    
    # Load pseudo labels
    logger.info(f"Loading pseudo labels from: {pseudo_file}")
    pseudo_df = pd.read_csv(pseudo_file, encoding="utf-8")
    
    # Check required columns
    if "tags" not in pseudo_df.columns:
        raise ValueError("Pseudo labels file must contain 'tags' column")
    
    if pseudo_id_col not in pseudo_df.columns:
        raise ValueError(f"Pseudo labels missing column: {pseudo_id_col}")
    
    if duvo_id_col not in duvo_df.columns:
        raise ValueError(f"DUVO file missing column: {duvo_id_col}")
    
    logger.info(f"DUVO data: {len(duvo_df)} rows")
    logger.info(f"Pseudo labels: {len(pseudo_df)} rows")
    
    # Prepare join keys
    pseudo_df["_join_key"] = pseudo_df[pseudo_id_col].astype(str).str.strip()
    duvo_df["_join_key"] = duvo_df[duvo_id_col].astype(str).str.strip()
    
    # Merge pseudo labels with DUVO data
    logger.info("Merging pseudo labels with DUVO data...")
    merged_df = duvo_df.merge(
        pseudo_df[["_join_key", "tags", "choice_id", "level", "confidence", "verified"]], 
        on="_join_key", 
        how="left"
    )
    
    # Clean up join key
    merged_df = merged_df.drop(columns=["_join_key"])
    
    # Create the Upload Product Template structure
    logger.info("Creating Upload Product Template format...")
    
    template_df = pd.DataFrame()
    
    # Tags column (first column as per template)
    template_df["Tags"] = merged_df["tags"].fillna("")
    
    # Product Type - use from DUVO or set default
    product_type_col = "نوع المنتج / Product Type"
    if product_type_col in merged_df.columns:
        template_df["Product Type "] = merged_df[product_type_col].fillna("Pet Supplies")
    else:
        template_df["Product Type "] = "Pet Supplies"
    
    # Title - use Arabic title from DUVO
    title_col = "product_title_ar"
    if title_col in merged_df.columns:
        template_df["Title"] = merged_df[title_col].fillna("")
    else:
        # Fallback to product name
        name_col = "اسم المنتج"
        if name_col in merged_df.columns:
            template_df["Title"] = merged_df[name_col].fillna("")
        else:
            template_df["Title"] = ""
    
    # Variant SKU
    sku_col = "كود المنتج - SKU"
    if sku_col in merged_df.columns:
        template_df["Variant SKU"] = merged_df[sku_col].fillna("")
    else:
        template_df["Variant SKU"] = ""
    
    # Variant Price
    price_col = "سعر البيع"
    if price_col in merged_df.columns:
        template_df["Variant Price"] = merged_df[price_col].fillna("")
    else:
        template_df["Variant Price"] = ""
    
    # Variant Cost
    cost_col = "سعر الجملة / Cost"
    if cost_col in merged_df.columns:
        template_df["Variant Cost"] = merged_df[cost_col].fillna("")
    else:
        template_df["Variant Cost"] = ""
    
    # Body (HTML) - use Arabic description
    desc_col = "arabic description"
    if desc_col in merged_df.columns:
        template_df["Body (HTML)"] = merged_df[desc_col].fillna("")
    else:
        # Fallback to regular description
        fallback_desc = "وصف المنتج"
        if fallback_desc in merged_df.columns:
            template_df["Body (HTML)"] = merged_df[fallback_desc].fillna("")
        else:
            template_df["Body (HTML)"] = ""
    
    # Vendor
    vendor_col = "الماركة"
    if vendor_col in merged_df.columns:
        template_df["Vendor"] = merged_df[vendor_col].fillna("")
    else:
        # Fallback to company name
        company_col = "اسم الشركة او البائع"
        if company_col in merged_df.columns:
            template_df["Vendor"] = merged_df[company_col].fillna("")
        else:
            template_df["Vendor"] = ""
    
    # Image Src
    img_col = "Image Src"
    if img_col in merged_df.columns:
        template_df["Image Src"] = merged_df[img_col].fillna("")
    else:
        template_df["Image Src"] = ""
    
    # Status
    template_df["Status"] = "active"
    
    # Published
    template_df["Published "] = True
    
    # Total Inventory Qty
    qty_col = "كمية المخزون لكل منتج"
    if qty_col in merged_df.columns:
        template_df["Total Inventory Qty"] = merged_df[qty_col].fillna(0)
    else:
        template_df["Total Inventory Qty"] = 0
    
    # Metafield: custom.brand
    brand_col = "الماركة"
    if brand_col in merged_df.columns:
        template_df["Metafield: custom.brand"] = merged_df[brand_col].fillna("")
    else:
        template_df["Metafield: custom.brand"] = ""
    
    # Preserve original fields as requested
    # meta_title - keep as is
    meta_title_col = "meta_title"
    if meta_title_col in merged_df.columns:
        template_df["meta_title"] = merged_df[meta_title_col]
    
    # meta_description - keep as is  
    meta_desc_col = "meta_description"
    if meta_desc_col in merged_df.columns:
        template_df["meta_description"] = merged_df[meta_desc_col]
    
    # product_title_ar - keep as is
    product_title_ar_col = "product_title_ar"
    if product_title_ar_col in merged_df.columns:
        template_df["product_title_ar"] = merged_df[product_title_ar_col]
    
    # Clean up empty tags
    template_df["Tags"] = template_df["Tags"].apply(safe_str)
    
    logger.info(f"Formatted data: {len(template_df)} rows")
    logger.info(f"Rows with tags: {len(template_df[template_df['Tags'].str.len() > 0])}")
    
    # Save to Excel
    logger.info(f"Saving to: {output_file}")
    template_df.to_excel(output_file, index=False, engine="openpyxl")
    
    logger.info("✅ Upload template with tags created successfully!")
    
    # Print summary
    print("\n=== DUVO UPLOAD TEMPLATE WITH TAGS ===")
    print(f"Output file: {output_file}")
    print(f"Total rows: {len(template_df)}")
    print(f"Rows with tags: {len(template_df[template_df['Tags'].str.len() > 0])}")
    print(f"Columns: {list(template_df.columns)}")
    
    # Show sample tags
    tagged_rows = template_df[template_df["Tags"].str.len() > 0]
    if len(tagged_rows) > 0:
        print("\nSample products with tags:")
        for i, (_, row) in enumerate(tagged_rows.head(5).iterrows()):
            print(f"  {i+1}. {row['Title'][:60]}...")
            print(f"     Tags: {row['Tags']}")
            print(f"     SKU: {row['Variant SKU']}")
            print(f"     Vendor: {row['Vendor']}")
            print()
    
    # Show products without tags
    untagged_rows = template_df[template_df["Tags"].str.len() == 0]
    if len(untagged_rows) > 0:
        print(f"\nProducts without tags: {len(untagged_rows)}")
        print("These products either:")
        print("- Were not processed by bootstrap_pseudolabels.py")
        print("- Had no matching SKU in pseudo labels")
        print("- Had confidence below threshold")
    
    return template_df

def main():
    parser = argparse.ArgumentParser(description="Append tags to DUVOtranslated.xlsx in Upload Template format")
    parser.add_argument("--duvo", default="DUVOtranslated.xlsx", help="DUVOtranslated.xlsx file")
    parser.add_argument("--pseudo", default="pseudo_labels_duvo.csv", help="pseudo_labels.csv file (with tags)")
    parser.add_argument("--output", default="DUVO_upload_template_with_tags.xlsx", help="Output Excel file")
    parser.add_argument("--pseudo-id-col", default="product_id", help="ID column in pseudo file")
    parser.add_argument("--duvo-id-col", default="كود المنتج - SKU", help="ID column in DUVO file")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    
    args = parser.parse_args()
    
    try:
        append_tags_to_duvo(
            duvo_file=args.duvo,
            pseudo_file=args.pseudo,
            output_file=args.output,
            pseudo_id_col=args.pseudo_id_col,
            duvo_id_col=args.duvo_id_col,
            log_level=args.log_level
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
