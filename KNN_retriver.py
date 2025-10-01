
"""
Build an embeddings index for taxonomy nodes (L3/L4) from descriptors.

Input  : taxonomy_descriptors.xlsx (sheet "descriptors" by default)
Output : node_embeddings.parquet (columns: node_id, level, path_en, text, dim, embedding)
         (optional) nodes.faiss + nodes.faiss.meta.json (if --faiss)

Demo   : --demo-query "Arabic or English text..." --topk 8

Usage (PowerShell/ CMD):
python KNN_retriver.py --inp "taxonomy_descriptors.xlsx" --sheet "descriptors" --out "node_embeddings.parquet" --model "text-embedding-3-large" --batch 64 --faiss --log-level INFO

"""
import math

import os, sys, re, json, argparse, logging, hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

try:
    import openai
except Exception as e:
    print("Please `pip install openai` (official SDK).", file=sys.stderr)
    raise

# Optional FAISS
try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

# ---------- Logging ----------
logger = logging.getLogger("gen_node_embeddings")

def setup_logger(level_str: str = "INFO"):
    logger.handlers.clear()
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(h)
    level = getattr(logging, level_str.upper(), logging.INFO)
    logger.setLevel(level)

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate embeddings for taxonomy descriptor nodes.")
    p.add_argument("--inp", required=True, help="Path to taxonomy_descriptors.xlsx (or .csv)")
    p.add_argument("--sheet", default="descriptors", help="Worksheet name if Excel (default: descriptors)")
    p.add_argument("--out", default="node_embeddings.parquet", help="Parquet file output path")
    p.add_argument("--model", default="text-embedding-3-large", help="Embedding model name")
    p.add_argument("--batch", type=int, default=64, help="Batch size for embedding calls")
    p.add_argument("--cache", default="emb_cache.jsonl", help="Cache file for text->vector")
    p.add_argument("--faiss", action="store_true", help="Also write FAISS index (requires faiss-cpu)")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    p.add_argument("--demo-query", default=None, help="Optional: run a demo retrieval with this query text")
    p.add_argument("--topk", type=int, default=8, help="Top-K results for demo retrieval")
    return p.parse_args()

# ---------- IO helpers ----------
def read_table(path: Path, sheet: str) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet)
    elif path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input extension: {path.suffix}")

def ensure_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")

# ---------- Text building ----------
def norm_space(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u00A0"," ").replace("\ufeff"," ")
    s = re.sub(r"\s+"," ", s).strip()
    return s

def build_path_en(row: pd.Series) -> str:
    parts = [row.get("L1_en"), row.get("L2_en"), row.get("L3_en")]
    if norm_space(row.get("L4_en")):
        parts.append(row.get("L4_en"))
    parts = [norm_space(p) for p in parts if norm_space(p)]
    return " > ".join(parts)

def build_text(row: pd.Series) -> str:
    # Bilingual, compact node text: path + EN + AR descriptors
    path = build_path_en(row)
    de = norm_space(row.get("descriptor_en"))
    da = norm_space(row.get("descriptor_ar"))
    return f"{path} :: {de} | {da}".strip()

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
    def get(self, key: str):
        return self.map.get(key)
    def put(self, key: str, rec: Dict[str, Any]):
        out = {"key": key, **rec}
        self.map[key] = out
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

def text_key(text: str, model: str) -> str:
    payload = json.dumps({"m": model, "t": text}, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

# ---------- OpenAI embeddings ----------
class TemporaryError(Exception): ...

@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.0, min=1, max=12),
    retry=retry_if_exception_type(TemporaryError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def embed_batch(client, model: str, texts: List[str]) -> List[List[float]]:
    try:
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]
    except Exception as e:
        # network/5xx/timeouts/rate limits → retry
        raise TemporaryError(str(e))

# ---------- Cosine helpers ----------
def l2norm(v: List[float]) -> List[float]:
    s = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/s for x in v]

def cosine(a: List[float], b: List[float]) -> float:
    return sum(x*y for x,y in zip(a,b)) / ((math.sqrt(sum(x*x for x in a)) or 1.0) * (math.sqrt(sum(x*x for x in b)) or 1.0))

# ---------- Main ----------
def main():
    args = parse_args()
    setup_logger(args.log_level)

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set.")
        sys.exit(1)

    inp = Path(args.inp)
    df = read_table(inp, args.sheet)
    needed = ["node_id","L1_en","L2_en","L3_en","L4_en","level","descriptor_en","descriptor_ar"]
    ensure_cols(df, needed)

    # Keep only L3/L4 nodes (should already be the case)
    df = df.copy()
    df["path_en"] = df.apply(build_path_en, axis=1)
    df["text"] = df.apply(build_text, axis=1)

    # Drop empties
    df = df[df["node_id"].notna() & df["text"].astype(str).str.strip().ne("")]
    df = df.drop_duplicates(subset=["node_id"]).reset_index(drop=True)

    logger.info("Nodes to embed: %d", len(df))

    cache = JsonlCache(Path(args.cache))
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError("No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
    openai.api_key = api_key
    client = openai

    # Build vectors
    batch = max(1, int(args.batch))
    embeddings: List[List[float]] = [None] * len(df)  # type: ignore

    idxs_to_embed: List[int] = []
    for i, row in df.iterrows():
        t = row["text"]
        key = text_key(t, args.model)
        c = cache.get(key)
        if c:
            embeddings[i] = c["vec"]
        else:
            idxs_to_embed.append(i)

    logger.info("Cached: %d | To compute: %d", len(df) - len(idxs_to_embed), len(idxs_to_embed))

    for start in range(0, len(idxs_to_embed), batch):
        chunk_idxs = idxs_to_embed[start:start+batch]
        texts = [df.loc[i, "text"] for i in chunk_idxs]
        logger.info("Embedding batch %d..%d / %d", start+1, min(start+batch, len(idxs_to_embed)), len(idxs_to_embed))
        vecs = embed_batch(client, args.model, texts)
        for i, v in zip(chunk_idxs, vecs):
            # normalize to unit length (recommended for cosine)
            v = l2norm(v)
            embeddings[i] = v
            key = text_key(df.loc[i, "text"], args.model)
            cache.put(key, {"vec": v, "dim": len(v)})

    # Assemble output
    dims = [len(v) if v is not None else 0 for v in embeddings]
    if any(d == 0 for d in dims):
        logger.error("Some embeddings failed. Please re-run; cache will fill gaps.")
        sys.exit(2)

    out_df = df[["node_id","level","path_en","text"]].copy()
    out_df["dim"] = dims
    out_df["embedding"] = embeddings

    out_path = Path(args.out)
    out_df.to_parquet(out_path, index=False)
    logger.info("Saved embeddings: %s (rows=%d, dim=%d)", str(out_path.resolve()), len(out_df), out_df["dim"].iloc[0])

    # Optional FAISS
    if args.faiss:
        if not FAISS_AVAILABLE:
            logger.warning("faiss-cpu not installed; skipping FAISS index.")
        else:
            import numpy as np
            X = np.array(out_df["embedding"].tolist(), dtype="float32")
            index = faiss.IndexFlatIP(X.shape[1])  # cosine with normalized vectors == inner product
            index.add(X)
            faiss.write_index(index, "nodes.faiss")
            meta = {
                "rows": len(out_df),
                "dim": int(X.shape[1]),
                "ids": out_df["node_id"].tolist(),
                "path_en": out_df["path_en"].tolist(),
            }
            Path("nodes.faiss.meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            logger.info("Saved FAISS index: nodes.faiss (+ nodes.faiss.meta.json)")

    # Demo retrieval (optional)
    if args.demo_query:
        logger.info("Running demo retrieval: %s", args.demo_query)
        # embed query
        q_vec = embed_batch(client, args.model, [args.demo_query])[0]
        q_vec = l2norm(q_vec)
        # cosine over pandas (no faiss required)
        import numpy as np
        M = np.array(out_df["embedding"].tolist(), dtype="float32")
        q = np.array(q_vec, dtype="float32")
        sims = M @ q  # since normalized
        topk = min(args.topk, len(out_df))
        idx = sims.argsort()[-topk:][::-1]
        logger.info("Top-%d matches:", topk)
        for rank, j in enumerate(idx, 1):
            logger.info("%2d) %.4f  %s  [%s]", rank, float(sims[j]), out_df.loc[j, "node_id"], out_df.loc[j, "path_en"])

if __name__ == "__main__":
    main()
