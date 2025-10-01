#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add tags column to existing pseudo_labels_duvo.csv file.

This script takes an existing pseudo_labels.csv file and adds the tags column
by looking up the hierarchical tags from the descriptors file.
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
    return logging.getLogger("add_tags")

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

def add_tags_to_pseudo(pseudo_file: str, descriptors_file: str, output_file: str, 
                      descriptors_sheet: str = "descriptors",
                      log_level: str = "INFO"):
    """
    Add tags column to existing pseudo labels file.
    
    Args:
        pseudo_file: Path to existing pseudo_labels.csv
        descriptors_file: Path to taxonomy_descriptors.xlsx
        output_file: Path to output CSV file with tags
        descriptors_sheet: Sheet name in descriptors file
        log_level: Logging level
    """
    logger = setup_logger(log_level)
    
    # Load pseudo labels (supports CSV and Excel)
    logger.info(f"Loading pseudo labels from: {pseudo_file}")
    pseudo_path = Path(pseudo_file)
    if pseudo_path.suffix.lower() in {".xlsx", ".xls"}:
        pseudo_df = pd.read_excel(pseudo_path)
    else:
        pseudo_df = pd.read_csv(pseudo_path, encoding="utf-8")
    
    # Load descriptors
    logger.info(f"Loading descriptors from: {descriptors_file}")
    desc_df = pd.read_excel(Path(descriptors_file), sheet_name=descriptors_sheet)
    
    # Check required columns
    if "choice_id" not in pseudo_df.columns:
        raise ValueError("Pseudo labels missing 'choice_id' column")
    
    need_desc = ["node_id", "level", "L1_en", "L2_en", "L3_en", "L4_en"]
    for c in need_desc:
        if c not in desc_df.columns:
            raise ValueError(f"Descriptors missing column: {c}")
    
    logger.info(f"Pseudo labels: {len(pseudo_df)} rows")
    logger.info(f"Descriptors: {len(desc_df)} rows")
    
    # Merge pseudo labels with descriptors to get hierarchical tags
    logger.info("Merging with descriptors to get hierarchical tags...")
    merged_df = pseudo_df.merge(
        desc_df[["node_id", "level", "L1_en", "L2_en", "L3_en", "L4_en"]], 
        left_on="choice_id", 
        right_on="node_id", 
        how="left"
    )
    
    # Build tags column
    logger.info("Building hierarchical tags...")
    merged_df["tags"] = merged_df.apply(build_hierarchical_tags, axis=1)
    
    # Clean up extra columns from merge
    output_df = pseudo_df.copy()
    output_df["tags"] = merged_df["tags"]
    
    # Clean up empty tags
    output_df["tags"] = output_df["tags"].apply(safe_str)
    
    logger.info(f"Tags added to {len(output_df)} rows")
    logger.info(f"Rows with tags: {len(output_df[output_df['tags'].str.len() > 0])}")
    
    # Save to file (CSV or Excel based on extension)
    logger.info(f"Saving to: {output_file}")
    output_path = Path(output_file)
    if output_path.suffix.lower() in {".xlsx", ".xls"}:
        output_df.to_excel(output_path, index=False)
    else:
        output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    logger.info("✅ Tags added successfully!")
    
    # Print summary
    print("\n=== TAGS ADDED TO PSEUDO LABELS ===")
    print(f"Output file: {output_file}")
    print(f"Total rows: {len(output_df)}")
    print(f"Rows with tags: {len(output_df[output_df['tags'].str.len() > 0])}")
    
    # Show sample tags
    tagged_rows = output_df[output_df["tags"].str.len() > 0]
    if len(tagged_rows) > 0:
        print("\nSample tags:")
        for i, (_, row) in enumerate(tagged_rows.head(5).iterrows()):
            print(f"  {i+1}. {row['title'][:60]}...")
            print(f"     Tags: {row['tags']}")
            print(f"     Choice ID: {row['choice_id']}")
            print()
    
    return output_df

def main():
    parser = argparse.ArgumentParser(description="Add tags column to existing pseudo labels")
    parser.add_argument("--pseudo", default="pseudo_labels_duvo.csv", help="Existing pseudo_labels.csv file")
    parser.add_argument("--descriptors", default="pets/taxonomy_descriptorsAnimals.xlsx", help="Descriptors Excel file")
    parser.add_argument("--sheet", default="descriptors", help="Descriptors sheet name")
    parser.add_argument("--output", default="pseudo_labels_duvo_with_tags.csv", help="Output CSV file")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    
    args = parser.parse_args()
    
    try:
        add_tags_to_pseudo(
            pseudo_file=args.pseudo,
            descriptors_file=args.descriptors,
            output_file=args.output,
            descriptors_sheet=args.sheet,
            log_level=args.log_level
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
