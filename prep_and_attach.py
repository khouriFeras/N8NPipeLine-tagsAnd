#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, sys, subprocess
from pathlib import Path
import pandas as pd

def normalize_pseudo(pseudo_path: Path) -> Path:
    df = pd.read_csv(pseudo_path)
    # Ensure the chosen node column the attach script expects
    if "choice_id" not in df.columns:
        if "node_id" in df.columns:
            df["choice_id"] = df["node_id"]
        else:
            raise SystemExit("Pseudo CSV needs either 'choice_id' or 'node_id' column.")

    # Ensure there is a clean 'Title' column to join on (copy from best available)
    for c in ["Title", "title", "product_id", "Handle", "handle", "name", "Name"]:
        if c in df.columns:
            df["Title"] = df[c].astype(str).str.strip()
            break
    else:
        raise SystemExit(f"No title-like column found in pseudo CSV. Columns: {list(df.columns)}")

    out = pseudo_path.with_name(pseudo_path.stem + "_EASY.csv")
    df.to_csv(out, index=False)
    return out

def normalize_products(products_path: Path, sheet: str) -> Path:
    df = pd.read_excel(products_path, sheet_name=sheet)
    if "Title" not in df.columns:
        raise SystemExit(f"'Title' column not found in {products_path} sheet '{sheet}'.")
    df["Title"] = df["Title"].astype(str).str.strip()
    out = products_path.with_name(products_path.stem + "_STRIPPED.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, sheet_name=sheet, index=False)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", required=True)
    ap.add_argument("--pseudo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--descriptors", default="taxonomy_descriptorsAnimals.xlsx")
    ap.add_argument("--sheet", default="descriptors")
    ap.add_argument("--prod-sheet", default="Sheet1")
    ap.add_argument("--attach", default="attach_L3_L4.py")  # path to your attach script
    args = ap.parse_args()

    products = Path(args.products)
    pseudo = Path(args.pseudo)
    descriptors = Path(args.descriptors)

    pseudo_easy = normalize_pseudo(pseudo)
    products_stripped = normalize_products(products, args.prod_sheet)

    # Join on Title↔Title so we avoid case/space mismatches
    cmd = [
        sys.executable, str(Path(args.attach)),
        "--products", str(products_stripped), "--prod-sheet", args.prod_sheet,
        "--pseudo", str(pseudo_easy),
        "--descriptors", str(descriptors), "--sheet", args.sheet,
        "--prod-id-col", "Title", "--pseudo-id-col", "Title",
        "--out", args.out, "--log-level", "INFO",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Done →", args.out)

if __name__ == "__main__":
    main()
