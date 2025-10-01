#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bootstrap high-confidence pseudo-labels for products using:
1) k-NN retrieval over taxonomy node embeddings
2) LLM constrained choice (choose one candidate ID or UNKNOWN)
3) Verification: literal evidence (+ optional semantic check)

Speedups in this version:
- Auto-accept strong top-1 with healthy margin (skip LLM)
- Adaptive K (4/6/base) to shrink prompts
- Trimmed descriptors and capped max_tokens
- Concurrent LLM calls (--workers)

Example:
  python bootstrap_pseudolabels.py ^
    --nodes "node_embeddings.parquet" ^
    --descriptors "Pets\\outPut\\taxonomy_descriptorsAnimals.xlsx" --sheet "descriptors" ^
    --products ""D:\JafarShop\tagsFinal\beapharStage2.xlsx"" --prod-sheet "Sheet1" ^
    --title-col "Title" --desc-col "description no html" ^
    --embed-model "text-embedding-3-large" ^
    --llm-model "gpt-5-mini" ^
    --topk 8 --conf-thresh 0.90 --verify-semantic ^
    --sim-ok 0.82 --margin-ok 0.10 --workers 12 ^
    --out "pseudo_labels.csv" --log-level DEBUG
    
    python bootstrap_pseudolabels.py --nodes "node_embeddings.parquet" --descriptors "Pets\\outPut\\taxonomy_descriptorsAnimals.xlsx" --sheet "descriptors" --products "DUVO\duvoFormatedSEO.xlsx" --prod-sheet "Sheet1" --title-col "Title" --desc-col "description no html" --embed-model "text-embedding-3-large" --llm-model "gpt-5-mini" --topk 8 --conf-thresh 0.90 --verify-semantic --sim-ok 0.82 --margin-ok 0.10 --workers 12 --out "pseudo_labels.csv" --log-level DEBUG

"""

import os, sys, re, json, argparse, logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

# ---------- Logging ----------
logger = logging.getLogger("bootstrap_labels")
def setup_logger(level_str: str = "INFO"):
    logger.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap pseudo-labels with retrieval + LLM + verification (fast).")
    p.add_argument("--nodes", required=True, help="node_embeddings.parquet")
    p.add_argument("--descriptors", required=True, help="taxonomy_descriptors.xlsx")
    p.add_argument("--sheet", default="descriptors", help="Sheet name for descriptors (default: descriptors)")
    p.add_argument("--products", required=True, help="Products file (.xlsx/.csv)")
    p.add_argument("--prod-sheet", default=None, help="Worksheet if products is Excel")
    p.add_argument("--title-col", default="Title", help="Product title column")
    p.add_argument("--desc-col", default="Cleaned_Body", help="Product description column")
    p.add_argument("--id-col", default=None, help="Optional product id/handle column to write through")

    p.add_argument("--embed-model", default="text-embedding-3-large", help="Embedding model for retrieval")
    p.add_argument("--llm-model", default="gpt-5-mini", help="Chat model for classification")

    p.add_argument("--topk", type=int, default=8, help="Base K candidates to send to LLM")
    p.add_argument("--conf-thresh", type=float, default=0.85, help="Min confidence to accept")
    p.add_argument("--verify-semantic", action="store_true", help="Embed evidence spans and verify semantically")
    p.add_argument("--sem-thresh", type=float, default=0.80, help="Semantic evidence similarity threshold")

    # Speed knobs
    p.add_argument("--sim-ok", type=float, default=0.82, help="Auto-accept if top1 similarity ≥ SIM_OK")
    p.add_argument("--margin-ok", type=float, default=0.10, help="Auto-accept if (top1 - top2) ≥ MARGIN_OK")
    p.add_argument("--adaptive", action="store_true", default=True, help="Enable adaptive-K (default on)")
    p.add_argument("--no-adaptive", dest="adaptive", action="store_false", help="Disable adaptive-K")
    p.add_argument("--max-cand-desc", type=int, default=120, help="Trim candidate descriptor_en to this many chars")
    p.add_argument("--max-tokens", type=int, default=80, help="Cap LLM completion tokens")
    p.add_argument("--workers", type=int, default=8, help="Parallel LLM calls")

    p.add_argument("--max-rows", type=int, default=0, help="Limit number of products (0 = all)")
    p.add_argument("--out", default="pseudo_labels.csv", help="Output CSV path")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    p.add_argument("--embed-item-max-toks", type=int, default=256,
               help="Truncate each product text to this many tokens before embedding")
    p.add_argument("--embed-batch-max-toks", type=int, default=270_000,
                   help="Max total tokens per embeddings.create call (API limit ~300k)")
    p.add_argument("--embed-batch-max-size", type=int, default=128,
               help="Max items per embeddings batch")

    return p.parse_args()

# ---------- Helpers ----------
def norm_space(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u00A0"," ").replace("\ufeff"," ")
    s = re.sub(r"\s+"," ", s).strip()
    return s

def l2norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return v / n

def build_hierarchical_tags(row) -> str:
    """Build hierarchical tags from L1_en, L2_en, L3_en, L4_en columns."""
    tags = []
    for level in ["L1_en", "L2_en", "L3_en", "L4_en"]:
        if level in row and pd.notna(row[level]) and str(row[level]).strip():
            tag = str(row[level]).strip()
            if tag.lower() not in {"", "nan", "none", "null", "n/a", "-"}:
                tags.append(tag)
    return ",".join(tags)
# --- tokenization helpers (tiktoken if available, else safe fallback) ---
try:
    import tiktoken
    _enc = tiktoken.get_encoding("o200k_base")
    def tok_len(s: str) -> int:
        return len(_enc.encode(s))
    def truncate_to_toks(s: str, max_toks: int) -> str:
        toks = _enc.encode(s)
        return _enc.decode(toks[:max_toks])
except Exception:
    def tok_len(s: str) -> int:
        # rough fallback: ~4 chars per token
        return max(1, len(s) // 4)
    def truncate_to_toks(s: str, max_toks: int) -> str:
        # rough fallback: approximate by chars
        return s[: max_toks * 4]

# ---------- OpenAI SDK ----------
try:
    import openai
except Exception:
    print("Please `pip install openai` (official SDK).", file=sys.stderr)
    raise

class TemporaryError(Exception): ...
class ParseError(Exception): ...

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.0, min=1, max=12),
    retry=retry_if_exception_type(TemporaryError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def embed_texts(client, model: str, texts: List[str]) -> List[List[float]]:
    try:
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]
    except Exception as e:
        raise TemporaryError(str(e))

def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _coerce_choice(data: dict | list) -> dict:
    # accept list → first dict
    if isinstance(data, list):
        data = next((x for x in data if isinstance(x, dict)), {})
    # accept envelope
    if isinstance(data, dict):
        for key in ("result","output","data"):
            if key in data and isinstance(data[key], dict):
                data = data[key]
    # defaults
    choice_id = str(data.get("choice_id", "UNKNOWN") or "UNKNOWN")
    level = str(data.get("level", "UNKNOWN") or "UNKNOWN").upper()
    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    ev = data.get("evidence_spans", [])
    if isinstance(ev, str): ev = [ev]
    elif not isinstance(ev, list): ev = []
    reason = str(data.get("reason", "") or "").strip()
    if level not in {"L3","L4","UNKNOWN"}: level = "UNKNOWN"
    confidence = max(0.0, min(1.0, confidence))
    return {
        "choice_id": choice_id,
        "level": level,
        "confidence": confidence,
        "evidence_spans": [str(x) for x in ev if str(x).strip()],
        "reason": reason,
    }

SYSTEM_MSG = (
    "You are a strict product tagger for an Arabic e-commerce taxonomy.\n"
    "Choose the SINGLE best tag ID from the given candidate list OR return 'UNKNOWN'.\n"
    "Rules:\n"
    "- Never invent tags. Choose only from candidates.\n"
    "- If evidence is weak or ambiguous, return 'UNKNOWN'.\n"
    "- Quote ONE short Arabic evidence span copied verbatim from the product text.\n"
    "- Prefer Level 4 only if clearly supported; otherwise choose Level 3.\n"
    "Return ONLY JSON with keys: choice_id, level, confidence, evidence_spans, reason.\n"
    "If unsure, set choice_id='UNKNOWN', level='UNKNOWN', confidence=0."
)

def build_user_prompt(product_text: str, candidates: List[Dict[str,str]], max_desc: int) -> str:
    # keep payload tight: id, path_en, trimmed descriptor_en
    clean = []
    for c in candidates:
        clean.append({
            "id": c["node_id"],
            "labels_en": c["path_en"],
            "descriptor_en": (c.get("descriptor_en","") or "")[:max_desc],
            "level": c["level"],
        })
    payload = {
        "PRODUCT": {"text": product_text[:600]},
        "CANDIDATES": clean,
        "SCHEMA": {
            "choice_id": "STRING or 'UNKNOWN'",
            "level": "L4 or L3 or 'UNKNOWN'",
            "confidence": "0..1",
            "evidence_spans": ["max 1 short span"],
            "reason": "≤ 8 words"
        }
    }
    return json.dumps(payload, ensure_ascii=False)

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.0, min=1, max=12),
    retry=retry_if_exception_type((TemporaryError, ParseError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def llm_choose(client, model: str, product_text: str, candidates: list[dict], max_desc: int, max_tokens: int) -> dict:
    """
    Robust chooser:
    - Tries max_completion_tokens then max_tokens (model-dependent)
    - Handles empty content safely
    - Extracts JSON substring if model adds prose
    - Falls back to UNKNOWN (no exception) to avoid retry storms
    """
    def build():
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user",   "content": build_user_prompt(product_text, candidates, max_desc)},
            ],
        }

    ladders = [
        {"response_format": {"type": "json_object"}, "max_completion_tokens": int(max_tokens)},
        {"response_format": {"type": "json_object"}},  # no cap
        {"max_completion_tokens": int(max_tokens)},
        {"max_tokens": int(max_tokens)},               # for older models
        {},                                            # plain, no format/caps
    ]

    last_err = None
    for extra in ladders:
        try:
            resp = client.chat.completions.create(**build(), **extra)
            content = resp.choices[0].message.content or ""
            content = strip_code_fences(content)
            text = content.strip()
            if not text:
                last_err = "empty content"
                continue

            # Try strict JSON first
            try:
                data = json.loads(text)
                return _coerce_choice(data)
            except Exception:
                # Try to salvage a JSON object from any prose
                m = re.search(r"\{.*\}", text, flags=re.S)
                if m:
                    try:
                        data = json.loads(m.group(0))
                        return _coerce_choice(data)
                    except Exception as e2:
                        last_err = f"salvage failed: {e2}"
                        continue
                last_err = "no JSON found"
                continue

        except Exception as e:
            # Param mismatches (e.g., max_tokens vs max_completion_tokens, response_format unsupported) → try next rung
            msg = str(e)
            last_err = msg
            if "unsupported" in msg.lower() or "parameter" in msg.lower() or "response_format" in msg.lower():
                continue
            # Other hard errors → break and return UNKNOWN
            break

    # Final fallback: don't raise; return UNKNOWN so pipeline continues
    return {
        "choice_id": "UNKNOWN",
        "level": "UNKNOWN",
        "confidence": 0.0,
        "evidence_spans": [],
        "reason": last_err or "llm empty/invalid",
    }

# ---------- Main ----------
def main():
    args = parse_args()
    setup_logger(args.log_level)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set."); sys.exit(1)

    # Load nodes + descriptors
    nodes = pd.read_parquet(Path(args.nodes))
    for c in ["node_id","level","path_en","text","embedding"]:
        if c not in nodes.columns:
            logger.error("Missing column in nodes parquet: %s", c); sys.exit(1)

    desc_df = pd.read_excel(Path(args.descriptors), sheet_name=args.sheet)
    need_desc = ["node_id","level","descriptor_en","descriptor_ar","L1_en","L2_en","L3_en","L4_en"]
    for c in need_desc:
        if c not in desc_df.columns:
            logger.error("Descriptors missing column: %s", c); sys.exit(1)

    nodes = nodes.merge(desc_df[["node_id","level","descriptor_en","descriptor_ar","L1_en","L2_en","L3_en","L4_en"]], on=["node_id","level"], how="left")

    # Normalize node embeddings
    X = np.array(nodes["embedding"].tolist(), dtype="float32")
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    # Load products
    prod_path = Path(args.products)
    if prod_path.suffix.lower() == ".xlsx":
        # Force openpyxl to avoid xlrd dependency for .xlsx. If the file is mislabeled
        # (actually .xls or CSV with .xlsx extension), try sensible fallbacks.
        try:
            prods = (
                pd.read_excel(prod_path, sheet_name=args.prod_sheet, engine="openpyxl")
                if args.prod_sheet
                else pd.read_excel(prod_path, engine="openpyxl")
            )
        except Exception as e:
            emsg = str(e)
            # Check magic bytes to infer real format
            try:
                with open(prod_path, "rb") as fh:
                    magic4 = fh.read(4)
            except Exception:
                magic4 = b""
            if magic4.startswith(b"\xD0\xCF\x11\xE0"):
                # Old .xls compound file
                try:
                    prods = (
                        pd.read_excel(prod_path, sheet_name=args.prod_sheet, engine="xlrd")
                        if args.prod_sheet
                        else pd.read_excel(prod_path, engine="xlrd")
                    )
                except Exception as e2:
                    logger.error(
                        "File has .xlsx extension but is an old .xls. Install xlrd (`pip install xlrd`) or convert to .xlsx. Error: %s",
                        str(e2),
                    ); sys.exit(1)
            else:
                # Try as CSV (misnamed text file)
                try:
                    prods = pd.read_csv(prod_path, encoding="utf-8")
                except Exception:
                    try:
                        prods = pd.read_csv(prod_path, encoding="utf-8", sep=None, engine="python")
                    except Exception:
                        logger.error(
                            "Failed to read products file '%s'. If it's not a real .xlsx, please convert to .xlsx or provide a .csv.",
                            str(prod_path),
                        ); sys.exit(1)
    elif prod_path.suffix.lower() == ".xls":
        # Old .xls requires xlrd. Provide helpful error if missing.
        try:
            prods = (
                pd.read_excel(prod_path, sheet_name=args.prod_sheet, engine="xlrd")
                if args.prod_sheet
                else pd.read_excel(prod_path, engine="xlrd")
            )
        except Exception as e:
            logger.error(
                "Reading .xls requires xlrd >= 2.0.1. Install it (`pip install xlrd`) or convert the file to .xlsx. Error: %s",
                str(e),
            ); sys.exit(1)
    elif prod_path.suffix.lower() == ".csv":
        prods = pd.read_csv(prod_path, encoding="utf-8")
    else:
        logger.error("Unsupported products file type: %s", prod_path.suffix); sys.exit(1)

    for col in [args.title_col, args.desc_col]:
        if col not in prods.columns:
            logger.error("Missing product column: %s", col); sys.exit(1)

    if args.max_rows and args.max_rows > 0:
        prods = prods.head(args.max_rows).copy()

    # Build product texts and dedupe for embedding cache
    texts: List[str] = []
    meta: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}
    uniq_texts: List[str] = []
    uniq_indices: List[int] = []

    for _, r in prods.iterrows():
        title = norm_space(r.get(args.title_col))
        desc  = norm_space(r.get(args.desc_col))[:400]
        pid   = r.get(args.id_col) if args.id_col and args.id_col in prods.columns else None
        t = f"{title} :: {desc}".strip()
        txt = t if t else title
        texts.append(txt)
        meta.append({"title": title, "desc": desc, "pid": pid})
        if txt not in seen:
            seen[txt] = len(uniq_texts)
            uniq_texts.append(txt)
            uniq_indices.append(len(texts)-1)

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
    client = openai.OpenAI(api_key=api_key)

    # Embed only unique texts
    logger.info("Embedding %d products (unique=%d)...", len(texts), len(uniq_texts))

    # 1) per-item truncate by tokens
    uniq_texts_trunc = [truncate_to_toks(t, int(args.embed_item_max_toks)) for t in uniq_texts]
    uniq_toklens = [tok_len(t) for t in uniq_texts_trunc]

    # 2) batch by total tokens and batch size
    max_batch_toks = int(args.embed_batch_max_toks)
    max_batch_sz   = int(args.embed_batch_max_size)

    uniq_vecs: List[np.ndarray] = []
    i = 0
    while i < len(uniq_texts_trunc):
        batch = []
        batch_toks = 0
        j = i
        while j < len(uniq_texts_trunc) and len(batch) < max_batch_sz:
            tlen = uniq_toklens[j]
            if batch and (batch_toks + tlen > max_batch_toks):
                break
            batch.append(uniq_texts_trunc[j])
            batch_toks += tlen
            j += 1

        # call API for this batch
        vecs = embed_texts(client, args.embed_model, batch)
        for v in vecs:
            uniq_vecs.append(l2norm(np.array(v, dtype="float32")))
        i = j

    # 3) map back to all rows via the seen[] index
    qvecs: List[np.ndarray] = [None] * len(texts)
    for pos, txt in enumerate(texts):
        qvecs[pos] = uniq_vecs[ seen[txt] ]

    # Map back to all rows
    qvecs: List[np.ndarray] = [None] * len(texts)
    for txt, iuniq in seen.items():
        # find all positions with this text
        # (we know first occurrence index from seen; scan for equal text to fill)
        for pos, t in enumerate(texts):
            if t is txt:  # same object (fast path) or use == fallback
                qvecs[pos] = uniq_vecs[iuniq]
        # fallback equality for any non-identical strings with same content
    for i, v in enumerate(qvecs):
        if v is None:
            qvecs[i] = uniq_vecs[seen[texts[i]]]

    out_rows: List[Dict[str, Any]] = []

    # --- Phase 1: retrieval + auto-accept; queue LLM tasks for the rest ---
    tasks = []
    for i, (m, q) in enumerate(zip(meta, qvecs), start=1):
        sims = X @ q
        # get top 10 for stable margin calculation
        Kprobe = min(10, len(nodes))
        idx = sims.argpartition(-Kprobe)[-Kprobe:]
        idx = idx[np.argsort(sims[idx])[::-1]]
        cand_frame = nodes.iloc[idx].copy()

        top1 = float(sims[idx[0]])
        top2 = float(sims[idx[1]]) if len(idx) > 1 else (top1 - 1.0)
        margin = float(top1 - top2)
        cand_ids_all = cand_frame["node_id"].tolist()

        # ----- AUTO-ACCEPT easy wins (skip LLM) -----
        if top1 >= float(args.sim_ok) and margin >= float(args.margin_ok):
            best = cand_frame.iloc[0]
            tags = build_hierarchical_tags(best)
            out_rows.append({
                "product_id": m["pid"],
                "title": m["title"],
                "choice_id": str(best["node_id"]),
                "level": str(best["level"]),
                "confidence": float(min(0.99, 0.85 + (top1 - args.sim_ok) * 1.5)),
                "evidence_spans": "[]",
                "reason": "auto-accept by margin",
                "verified": 1,
                "retrieval_top1": top1,
                "retrieval_margin": margin,
                "cand_ids": json.dumps(cand_ids_all, ensure_ascii=False),
                "tags": tags,
            })
            continue
        # -------------------------------------------

        # Adaptive K (smaller list for fairly easy items)
        Kbase = max(2, int(args.topk))
        if args.adaptive:
            if top1 >= 0.85 and margin >= 0.10:
                K = min(4, Kbase)
            elif top1 >= 0.82 and margin >= 0.08:
                K = min(6, Kbase)
            else:
                K = Kbase
        else:
            K = Kbase

        # Build the exact K candidates (top-K from the probe set)
        cand_frame = cand_frame.head(K)
        cand_dicts = [
            {
                "node_id": row["node_id"],
                "path_en": row["path_en"],
                "level": row["level"],
                "descriptor_en": (row.get("descriptor_en","") or ""),
                "descriptor_ar": (row.get("descriptor_ar","") or ""),
            } for _, row in cand_frame.iterrows()
        ]

        tasks.append({
            "idx": i,
            "meta": m,
            "product_text": f"{m['title']} :: {m['desc']}",
            "cand_ids_all": cand_ids_all,
            "cand_dicts": cand_dicts,
            "top1": top1,
            "margin": margin,
        })

        if i % 50 == 0:
            logger.info("Queued %d items for LLM; auto-accepted so far: %d", len(tasks), len(out_rows))

    # --- Phase 2: run LLM on queued tasks (concurrently) ---
    def run_one(payload):
        m = payload["meta"]
        try:
            choice = llm_choose(client, args.llm_model, payload["product_text"], payload["cand_dicts"], args.max_cand_desc, args.max_tokens)
        except Exception as e:
            logger.warning("LLM failed on '%s': %s", m["title"][:60], str(e)[:200])
            choice = {"choice_id":"UNKNOWN","level":"UNKNOWN","confidence":0.0,"evidence_spans":[],"reason":"error"}

        # literal evidence check
        ev_lit = False
        text_norm = payload["product_text"]
        for span in choice.get("evidence_spans", []):
            s = str(span).strip()
            if s and s in text_norm:
                ev_lit = True
                break

        # optional semantic evidence (span vs chosen node vector)
        ev_sem = True
        if args.verify_semantic and choice.get("choice_id") and choice["choice_id"] != "UNKNOWN":
            try:
                chosen_rows = nodes[nodes["node_id"] == choice["choice_id"]]
                if len(chosen_rows):
                    node_vec = X[chosen_rows.index[0]]
                    spans = [s for s in choice.get("evidence_spans", []) if isinstance(s, str) and s.strip()]
                    if spans:
                        vecs = embed_texts(client, args.embed_model, spans)
                        sims_ev = []
                        for sv in vecs:
                            sv = l2norm(np.array(sv, dtype="float32"))
                            sims_ev.append(float(np.dot(node_vec, sv)))
                        ev_sem = max(sims_ev) >= float(args.sem_thresh)
                    else:
                        ev_sem = False
                else:
                    ev_sem = False
            except Exception as e:
                logger.warning("Semantic verify failed: %s", str(e)[:200])
                ev_sem = False

        verified = (choice.get("confidence",0.0) >= float(args.conf_thresh)) and ev_lit and (ev_sem if args.verify_semantic else True)

        # Get tags for the chosen node
        tags = ""
        choice_id = choice.get("choice_id","UNKNOWN")
        if choice_id != "UNKNOWN":
            # Find the chosen node in the candidates to get its tags
            chosen_node = None
            for cand in payload["cand_dicts"]:
                if str(cand["node_id"]) == str(choice_id):
                    chosen_node = cand
                    break
            
            if chosen_node:
                # Get the full node data from the nodes dataframe
                node_rows = nodes[nodes["node_id"] == choice_id]
                if len(node_rows) > 0:
                    tags = build_hierarchical_tags(node_rows.iloc[0])
                else:
                    # Fallback: try to build tags from candidate data if available
                    tags = build_hierarchical_tags(chosen_node)

        return {
            "product_id": m["pid"],
            "title": m["title"],
            "choice_id": choice_id,
            "level": choice.get("level","UNKNOWN"),
            "confidence": choice.get("confidence",0.0),
            "evidence_spans": json.dumps(choice.get("evidence_spans", []), ensure_ascii=False),
            "reason": choice.get("reason",""),
            "verified": int(verified),
            "retrieval_top1": payload["top1"],
            "retrieval_margin": payload["margin"],
            "cand_ids": json.dumps(payload["cand_ids_all"], ensure_ascii=False),
            "tags": tags,
        }

    if tasks:
        W = max(1, int(args.workers))
        logger.info("LLM running on %d items with %d workers...", len(tasks), W)
        with ThreadPoolExecutor(max_workers=W) as ex:
            futures = [ex.submit(run_one, t) for t in tasks]
            for j, fut in enumerate(as_completed(futures), start=1):
                out_rows.append(fut.result())
                if j % 50 == 0:
                    logger.info("LLM completed %d / %d", j, len(tasks))

    # Save
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(Path(args.out), index=False, encoding="utf-8-sig")

    accepted = int((out_df["choice_id"] != "UNKNOWN").sum()) if "choice_id" in out_df.columns else 0
    verified = int((out_df.get("verified", pd.Series(dtype=int)) == 1).sum()) if "verified" in out_df.columns else 0
    logger.info("Saved pseudo labels: %s (rows=%d, chosen=%d, verified=%d)",
                str(Path(args.out).resolve()), len(out_df), accepted, verified)

if __name__ == "__main__":
    main()
