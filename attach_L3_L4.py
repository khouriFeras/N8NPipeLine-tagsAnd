#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Attach L3 (always) and L4 (optional) to products using:
- pseudo_labels.csv (from bootstrap_pseudolabels.py)
- taxonomy_descriptors.xlsx (must contain node_id, level, L3_en, L4_en)
- products file (xlsx/csv)

This version does NOT create/keep L3_en/L4_en columns in the output.
It computes final L3/L4 via descriptor lookups and candidates only.
"""

import argparse, logging, sys, json, re
from pathlib import Path
from typing import List, Any, Optional
import pandas as pd

# ---------- utils ----------
def safe_str(v) -> str:
    s = "" if v is None else str(v).strip()
    return "" if s.lower() in {"", "nan", "none", "null", "n/a", "-"} else s

def norm_level(v) -> str:
    s = safe_str(v).upper()
    if s in {"L3", "L4"}: return s
    try:
        n = int(float(s))
        return f"L{n}" if n in (3,4) else s
    except Exception:
        return s

def parse_json_list(val: Any) -> Optional[list]:
    if not isinstance(val, str) or not val.strip(): return None
    s = val.strip()
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            return None

def parent_l3_id(node_id: str) -> Optional[str]:
    nid = safe_str(node_id)
    if not nid: return None
    parts = nid.split("/")
    return "/".join(parts[:3]) if len(parts) >= 3 else None

def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        c_new = re.sub(r"\s+", " ", c).strip()
        c_new = c_new.replace(" _", "_").replace("_ ", "_")
        ren[c] = c_new
    return df.rename(columns=ren)

# ---------- logging ----------
logger = logging.getLogger("attach_l3_l4")
def setup_logger(level: str = "INFO"):
    logger.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

# ---------- cli ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attach L3 (mandatory) and L4 (optional) to products.")
    p.add_argument("--products", required=True, help="Products file (.xlsx or .csv)")
    p.add_argument("--prod-sheet", default=None, help="Sheet name if Excel")
    p.add_argument("--pseudo", required=True, help="pseudo_labels.csv")
    p.add_argument("--descriptors", required=True, help="taxonomy_descriptors.xlsx")
    p.add_argument("--sheet", default=None, help="Descriptors sheet name (omit to use first sheet)")
    p.add_argument("--prod-id-col", default="Title", help="Join key in products (e.g., Handle or Title)")
    p.add_argument("--pseudo-id-col", default="title", help="Join key in pseudo (e.g., product_id or title)")
    p.add_argument("--verified-only", action="store_true", help="Use only pseudo rows with verified==1")
    p.add_argument("--out", default="sample_with_L3_L4.xlsx", help="Output path (.xlsx or .csv)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()

# ---------- io helpers ----------
def read_table(path: Path, sheet: Optional[str] = None) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet) if sheet else pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

def ensure_cols(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{name}: missing columns: {missing}. Available: {list(df.columns)}")

# ---------- main ----------
def main():
    args = parse_args()
    setup_logger(args.log_level)

    prod_path = Path(args.products)
    pseudo_path = Path(args.pseudo)
    desc_path = Path(args.descriptors)

    products = normalize_colnames(read_table(prod_path, args.prod_sheet))
    pseudo   = normalize_colnames(read_table(pseudo_path))
    desc     = normalize_colnames(read_table(desc_path, args.sheet))

    # descriptors required columns (used for lookup; not kept in output)
    need_desc = ["node_id", "level", "L3_en", "L4_en"]
    ensure_cols(desc, need_desc, "descriptors")
    desc["node_id"] = desc["node_id"].astype(str).str.strip()
    desc["level"]   = desc["level"].apply(norm_level)
    desc["L3_en"]   = desc["L3_en"].apply(safe_str)
    desc["L4_en"]   = desc["L4_en"].apply(safe_str)

    # pseudo essentials
    ensure_cols(pseudo, ["choice_id"], "pseudo")
    pseudo["choice_id"] = pseudo["choice_id"].astype(str).str.strip()
    if "level" in pseudo.columns:
        pseudo["level"] = pseudo["level"].apply(norm_level)
    if "verified" in pseudo.columns and args.verified_only:
        before = len(pseudo)
        pseudo = pseudo[pseudo["verified"] == 1].copy()
        logger.info("Verified-only: kept %d of %d pseudo rows", len(pseudo), before)

    # join keys
    ensure_cols(products, [args.prod_id_col], "products")
    ensure_cols(pseudo,   [args.pseudo_id_col], "pseudo")
    products["_join_key"] = products[args.prod_id_col].astype(str).str.strip()
    pseudo["_join_key"]   = pseudo[args.pseudo_id_col].astype(str).str.strip()

    # dedupe pseudo: best per product (verified desc, confidence desc)
    if "confidence" not in pseudo.columns: pseudo["confidence"] = 0.0
    if "verified" not in pseudo.columns:   pseudo["verified"] = 0
    pseudo_sorted = pseudo.sort_values(
        by=["_join_key", "verified", "confidence"],
        ascending=[True, False, False]
    )
    pseudo_best = pseudo_sorted.drop_duplicates(subset=["_join_key"], keep="first").copy()

    # Build descriptor index for lookups (no L3_en/L4_en columns will be emitted)
    desc_idx = desc.set_index("node_id")[["level", "L3_en", "L4_en"]].to_dict(orient="index")

    # Merge products with minimal pseudo columns only (NO L3_en/L4_en merged)
    keep_cols = ["_join_key", "choice_id", "level", "confidence", "verified"]
    if "cand_ids" in pseudo_best.columns:
        keep_cols.append("cand_ids")
    tagged = products.merge(pseudo_best[keep_cols], on="_join_key", how="left").drop(columns=["_join_key"])

    # init output columns (final only)
    tagged["L3"] = ""
    tagged["L4"] = ""

    # fill from chosen node
    def fill_from_choice(row):
        if safe_str(row.get("L3")):  # already set
            return row
        cid = safe_str(row.get("choice_id"))
        if not cid:
            return row
        info = desc_idx.get(cid)
        if info:
            lvl  = norm_level(info.get("level"))
            l3en = safe_str(info.get("L3_en"))
            l4en = safe_str(info.get("L4_en"))
            if lvl == "L3" and l3en:
                row["L3"] = l3en
                return row
            if lvl == "L4":
                if l3en:
                    row["L3"] = l3en
                if not safe_str(row.get("L4")) and l4en:
                    row["L4"] = l4en
                if not safe_str(row.get("L3")):
                    par = parent_l3_id(cid)
                    if par and par in desc_idx:
                        row["L3"] = safe_str(desc_idx[par].get("L3_en"))
                return row
        # try parent directly
        par = parent_l3_id(cid)
        if par and par in desc_idx:
            row["L3"] = safe_str(desc_idx[par].get("L3_en"))
        return row

    # fallback via candidate IDs (first viable)
    def fill_from_candidates(row):
        if safe_str(row.get("L3")):
            return row
        cands = parse_json_list(row.get("cand_ids"))
        if not cands:
            return row
        for nid in cands:
            nid = safe_str(nid)
            if not nid: continue
            info = desc_idx.get(nid)
            if info:
                lvl  = norm_level(info.get("level"))
                l3en = safe_str(info.get("L3_en"))
                if lvl == "L3" and l3en:
                    row["L3"] = l3en
                    break
                if lvl == "L4":
                    par = parent_l3_id(nid)
                    if par and par in desc_idx:
                        l3en2 = safe_str(desc_idx[par].get("L3_en"))
                        if l3en2:
                            row["L3"] = l3en2
                            break
            else:
                par = parent_l3_id(nid)
                if par and par in desc_idx:
                    l3en2 = safe_str(desc_idx[par].get("L3_en"))
                    if l3en2:
                        row["L3"] = l3en2
                        break
        return row

    tagged = tagged.apply(fill_from_choice, axis=1)
    if "cand_ids" in tagged.columns:
        tagged = tagged.apply(fill_from_candidates, axis=1)

    # clean
    tagged["L3"] = tagged["L3"].apply(safe_str)
    tagged["L4"] = tagged["L4"].apply(safe_str)
    total = len(tagged)
    have_l3 = int((tagged["L3"].astype(str).str.len() > 0).sum())
    have_l4 = int((tagged["L4"].astype(str).str.len() > 0).sum())
    logger.info("Rows: %d | with L3: %d (%.1f%%) | with L4: %d (%.1f%%)",
                total, have_l3, 100.0 * have_l3 / max(1,total),
                have_l4, 100.0 * have_l4 / max(1,total))
    out_path = Path(args.out)
    if out_path.suffix.lower() in [".xlsx", ".xls"]:
        tagged.to_excel(out_path, index=False)
    else:
        tagged.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Saved: %s", str(out_path.resolve()))

if __name__ == "__main__":
    main()
