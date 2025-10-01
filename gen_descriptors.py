
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate bilingual (EN+AR) descriptors for L3/L4 taxonomy nodes via OpenAI.

Usage (PowerShell/CMD):
python gen_descriptors.py --inp "Pets\TexoAnimal.XLSX" --sheet "Tags" --l1 "L1 Tags" --l2 "L2 Tags" --l3 "L3 Tags" --l4 "L4 Tags" --out "Pets\outPut\taxonomy_descriptorsAnimals.xlsx" --model "o4-mini" --batch 16 --log-level INFO
Notes:
- Header normalization is applied, so trailing/odd spaces are OK.
- For o4-mini, temperature is auto-omitted; if JSON mode unsupported, we fallback to plain mode.
"""

import os, sys, json, math, argparse, hashlib, re, logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

# ---------- OpenAI SDK ----------
try:
    import openai
except Exception as e:
    print("Please `pip install openai` (official SDK).", file=sys.stderr)
    raise

# ---------- Logging ----------
EMPTY_TOKENS = {"", "nan", "none", "null", "na", "n/a", "-"}

def cell_str(row: pd.Series, col: str) -> str:
    v = row.get(col)
    # Treat real NaN/None/empty as ""
    if v is None:
        return ""
    # Pandas NaN (float)
    try:
        import math
        if isinstance(v, float) and math.isnan(v):
            return ""
    except Exception:
        pass
    s = str(v).strip()
    return "" if s.lower() in EMPTY_TOKENS else s

logger = logging.getLogger("gen_descriptors")

def setup_logger(level_str: str = "INFO"):
    logger.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate EN+AR descriptors for taxonomy nodes.")
    p.add_argument("--inp", required=True, help="Input taxonomy Excel path")
    p.add_argument("--sheet", default="Tags", help="Worksheet name (default: Tags)")
    p.add_argument("--l1", default="L1 Tag (English)", help="L1 English column name")
    p.add_argument("--l2", default="L2 Tag (English)", help="L2 English column name")
    p.add_argument("--l3", default="L3 Tag (English)", help="L3 English column name")
    p.add_argument("--l4", default="L4 Tag (English)", help="L4 English column name")
    p.add_argument("--out", default="taxonomy_descriptors.xlsx", help="Output Excel path")
    p.add_argument("--model", default="o4-mini", help="Chat model (default: o4-mini)")
    p.add_argument("--batch", type=int, default=12, help="Nodes per API call (default: 12)")
    p.add_argument("--cache", default="descriptor_cache.jsonl", help="JSONL cache path")
    p.add_argument("--dry-run", action="store_true", help="Build rows but skip API")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()

# ---------- Header normalization ----------
def _norm_col(s: Any) -> str:
    s = str(s or "")
    # replace NBSP and BOM, collapse whitespace, strip, lowercase for matching
    s = s.replace("\u00A0"," ").replace("\ufeff"," ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

# ---------- Stable slugs ----------
try:
    # python-slugify
    from slugify import slugify as _slugify
except Exception:
    _slugify = None

def _to_slug(s: str) -> str:
    s = str(s or "").strip()
    if not s:
        return ""
    if _slugify:
        try:
            return _slugify(s, lowercase=True)  # python-slugify API
        except TypeError:
            return _slugify(s).lower()
    # fallback: simple slug
    s = re.sub(r"[^A-Za-z0-9]+", "-", s.strip().lower())
    return s.strip("-")

def stable_id(*parts: str) -> str:
    parts = [p for p in parts if isinstance(p, str) and p.strip()]
    slugs = [_to_slug(p) for p in parts]
    return "/".join([s for s in slugs if s])

# ---------- Cache ----------
class JsonlCache:
    def __init__(self, path: Path):
        self.path = path
        self.map: Dict[str, Dict[str, Any]] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        self.map[obj["key"]] = obj
                    except Exception:
                        pass

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.map.get(key)

    def put(self, key: str, value: Dict[str, Any]) -> None:
        rec = {"key": key, **value}
        self.map[key] = rec
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def cache_key(L1: str, L2: str, L3: str, L4: str) -> str:
    payload = json.dumps({"L1": L1, "L2": L2, "L3": L3, "L4": L4}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

# ---------- Prompts ----------
SYSTEM_MSG = """You generate short bilingual descriptors for product taxonomy nodes.
For each node, produce two fields:
- descriptor_en: 1–2 sentences (<=25 words), clear and specific, in English.
- descriptor_ar: 1–2 sentences (<=25 words), Modern Standard Arabic, concise and specific.
Focus on what it is, typical use, common features or units; avoid brand names.
Return STRICT JSON as an array of objects aligned by 'idx': [{"idx":0, "descriptor_en":"...", "descriptor_ar":"..."}, ...].
"""

def build_user_batch_payload(nodes: List[Dict[str, str]]) -> str:
    # Compact JSON payload; the model will mirror back a JSON array with idx/descriptor_en/descriptor_ar
    template = {
        "task": "Write bilingual descriptors for these taxonomy nodes.",
        "nodes": [
            {
                "idx": i,
                "L1_en": n["L1_en"],
                "L2_en": n["L2_en"],
                "L3_en": n["L3_en"],
                "L4_en": n["L4_en"],
                "level": n["level"],
            }
            for i, n in enumerate(nodes)
        ],
        "schema": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["idx", "descriptor_en", "descriptor_ar"],
                "properties": {
                    "idx": {"type": "integer"},
                    "descriptor_en": {"type": "string", "maxLength": 220},
                    "descriptor_ar": {"type": "string", "maxLength": 220}
                }
            }
        }
    }
    return json.dumps(template, ensure_ascii=False)

# ---------- Errors ----------
class TemporaryError(Exception): ...
class ParseError(Exception): ...

def _truncate(s: str, n: int = 600) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[:n] + "…"

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _parse_json_array(payload: str) -> List[Dict[str, Any]]:
    payload = _strip_code_fences(payload)
    data = json.loads(payload)

    # Accept an array or an object with 'items'/'results'
    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            data = data["items"]
        elif "results" in data and isinstance(data["results"], list):
            data = data["results"]

    if not isinstance(data, list):
        raise ParseError("Model did not return a JSON array.")

    out: List[Dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ParseError(f"Item {i} is not a JSON object.")
        for k in ("idx", "descriptor_en", "descriptor_ar"):
            if k not in item:
                raise ParseError(f"Missing key '{k}' in item {i}.")
        out.append({
            "idx": int(item["idx"]),
            "descriptor_en": str(item["descriptor_en"]).strip(),
            "descriptor_ar": str(item["descriptor_ar"]).strip(),
        })
    return out

# ---------- OpenAI call with fallbacks ----------
@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.0, min=1, max=12),
    retry=retry_if_exception_type((TemporaryError, ParseError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def call_openai_batch(client, model: str, nodes: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Single API call per batch of nodes.
    Tries: (temp+JSON) -> (JSON only) -> (plain).
    Returns list of {"idx","descriptor_en","descriptor_ar"}.
    """
    logger.info("OpenAI batch → model=%s, nodes=%d", model, len(nodes))

    user_payload = build_user_batch_payload(nodes)
    logger.debug("User payload (truncated): %s", _truncate(user_payload))

    base_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": user_payload},
        ],
    }

    attempts = [
        ("temp+json", dict(response_format={"type": "json_object"}, temperature=0.1)),
        ("json",      dict(response_format={"type": "json_object"})),
        ("plain",     dict()),
    ]

    last_err = None
    for label, extra in attempts:
        try:
            logger.debug("Calling chat.completions (%s)", label)
            resp = client.chat.completions.create(**base_kwargs, **extra)

            # Token usage (if available)
            try:
                u = getattr(resp, "usage", None)
                if u:
                    logger.info("Usage: prompt=%s, completion=%s, total=%s",
                                getattr(u, "prompt_tokens", "?"),
                                getattr(u, "completion_tokens", "?"),
                                getattr(u, "total_tokens", "?"))
            except Exception:
                pass

            content = resp.choices[0].message.content
            # Guard against models returning None/empty content (e.g., tool calls or filtered output)
            if content is None or (isinstance(content, str) and content.strip() == ""):
                raise ParseError("Empty content from model.")
            logger.debug("Raw model content (truncated): %s", _truncate(content, 800))
            return _parse_json_array(content)

        except Exception as e:
            emsg = str(e)
            last_err = emsg
            # Recognize common constraints to advance the fallback ladder
            if "temperature" in emsg.lower() and "unsupported" in emsg.lower():
                logger.warning("Temperature unsupported; retrying without it.")
                continue
            if "response_format" in emsg.lower() and "unsupported" in emsg.lower():
                logger.warning("JSON mode unsupported; retrying in plain mode.")
                continue
            logger.warning("Attempt '%s' failed: %s", label, _truncate(emsg, 400))
            # For other errors, still try next fallback
            continue

    # If all attempts failed, trigger retry via tenacity
    logger.error("All attempts failed. Last error: %s", _truncate(last_err, 500))
    raise TemporaryError(last_err or "Unknown error")

# ---------- Main ----------
def main():
    args = parse_args()
    setup_logger(args.log_level)

    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set. Set it and rerun.")
        sys.exit(1)

    # Read and normalize headers
    try:
        df = pd.read_excel(args.inp, sheet_name=args.sheet)
    except Exception as e:
        logger.error("Failed to read Excel '%s' sheet '%s': %s", args.inp, args.sheet, e)
        sys.exit(1)

    orig_cols = list(df.columns)
    df.columns = [_norm_col(c) for c in df.columns]
    l1 = _norm_col(args.l1); l2 = _norm_col(args.l2); l3 = _norm_col(args.l3); l4 = _norm_col(args.l4)

    # Map original->normalized for info
    logger.debug("Original headers: %s", orig_cols)
    logger.info("Using columns (normalized): L1='%s' L2='%s' L3='%s' L4='%s'", l1, l2, l3, l4)

    for col in (l1, l2, l3, l4):
        if col not in df.columns:
            logger.error("Column not found after normalization: %s", col)
            sys.exit(1)

    # Build L3/L4 node list with stable IDs, deduped
    rows: List[Dict[str, Any]] = []
    seen_ids = set()
    skipped_blank = skipped_dupe = 0
    for _, r in df.iterrows():
        L1 = cell_str(r, l1)
        L2 = cell_str(r, l2)
        L3 = cell_str(r, l3)
        L4 = cell_str(r, l4)
    
        # Only keep L4 if it truly exists; else fall back to L3; else skip
        if L4:
            level = "L4"
        elif L3:
            level = "L3"
        else:
            skipped_blank += 1
            continue
        
        node_id = stable_id(L1, L2, L3, L4)
        if not node_id:
            skipped_blank += 1
            continue
        
        if node_id in seen_ids:
            skipped_dupe += 1
            continue
        seen_ids.add(node_id)
    
        rows.append({
            "node_id": node_id,
            "L1_en": L1, "L2_en": L2, "L3_en": L3, "L4_en": L4,
            "level": level
        })

    logger.info("Unique nodes to process: %d (skipped blank=%d, dupes=%d)", len(rows), skipped_blank, skipped_dupe)

    cache = JsonlCache(Path(args.cache))
    if not args.dry_run:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise RuntimeError("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
        openai.api_key = api_key
        client = openai
    else:
        client = None

    out_records: List[Dict[str, Any]] = []
    B = max(1, int(args.batch))

    # Batch processing
    for start in range(0, len(rows), B):
        batch = rows[start:start+B]
        logger.info("Batch %d..%d of %d", start+1, min(start+B, len(rows)), len(rows))

        # Use cache where possible
        payload_for_api: List[Dict[str, Any]] = []
        idx_map: Dict[int, int] = {}  # local idx -> out_records index

        for i, node in enumerate(batch):
            key = cache_key(node["L1_en"], node["L2_en"], node["L3_en"], node["L4_en"])
            c = cache.get(key)
            if c:
                out_records.append({
                    **node,
                    "descriptor_en": c["descriptor_en"],
                    "descriptor_ar": c["descriptor_ar"]
                })
            else:
                payload_for_api.append({**node})
                idx_map[len(payload_for_api) - 1] = len(out_records)
                out_records.append({**node, "descriptor_en": None, "descriptor_ar": None})

        # If dry-run or all cached, skip API
        if args.dry_run or len(payload_for_api) == 0:
            continue

        # Minimal labels for API (don’t send node_id)
        api_nodes = [
            {"L1_en": n["L1_en"], "L2_en": n["L2_en"], "L3_en": n["L3_en"], "L4_en": n["L4_en"], "level": n["level"]}
            for n in payload_for_api
        ]

        # Call OpenAI
        results = call_openai_batch(client, args.model, api_nodes)

        # Place results and cache
        for item in results:
            local_idx = item["idx"]
            if local_idx not in idx_map:
                continue
            global_pos = idx_map[local_idx]
            desc_en = item["descriptor_en"].strip()
            desc_ar = item["descriptor_ar"].strip()

            out_records[global_pos]["descriptor_en"] = desc_en
            out_records[global_pos]["descriptor_ar"] = desc_ar

            n = payload_for_api[local_idx]
            key = cache_key(n["L1_en"], n["L2_en"], n["L3_en"], n["L4_en"])
            cache.put(key, {"descriptor_en": desc_en, "descriptor_ar": desc_ar})

    # Finalize
    out_df = pd.DataFrame(out_records)
    out_df = out_df.dropna(subset=["descriptor_en", "descriptor_ar"])

    out_path = Path(args.out)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        out_df.to_excel(w, index=False, sheet_name="descriptors")

    logger.info("Saved: %s  (rows=%d)", str(out_path.resolve()), len(out_df))

if __name__ == "__main__":
    main()
