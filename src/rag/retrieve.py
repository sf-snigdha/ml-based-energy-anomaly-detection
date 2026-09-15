from pathlib import Path
import argparse, json
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
IDX = ROOT / "data" / "rag_index"

config = json.loads((IDX / "config.json").read_text(encoding="utf-8"))
records = json.loads((IDX / "chunks.json").read_text(encoding="utf-8"))
emb = np.load(IDX / "embeddings.npy")
model = SentenceTransformer(config["model"])

p = argparse.ArgumentParser()
p.add_argument("query")
p.add_argument("--top-k", type=int, default=3)
args = p.parse_args()

q = model.encode([args.query], normalize_embeddings=True).astype("float32")[0]
scores = emb @ q
order = np.argsort(scores)[::-1][:args.top_k]

print(f"QUERY: {args.query}")
for rank, idx in enumerate(order, 1):
    r = records[int(idx)]
    print("\n" + "="*80)
    print(f"Rank {rank} | score={scores[idx]:.4f} | {r['source']}#{r['chunk_id']}")
    print(r["text"])
