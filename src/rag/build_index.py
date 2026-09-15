from pathlib import Path
import json, re
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "knowledge"
OUT = ROOT / "data" / "rag_index"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def chunks(text, size=900, overlap=150):
    text = re.sub(r"\\n{3,}", "\\n\\n", text.strip())
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return out

records = []
for path in sorted(DOCS.glob("*.md")):
    for i, chunk in enumerate(chunks(path.read_text(encoding="utf-8"))):
        records.append({"source": path.name, "chunk_id": i, "text": chunk})

model = SentenceTransformer(MODEL)
emb = model.encode([r["text"] for r in records], normalize_embeddings=True).astype("float32")

OUT.mkdir(parents=True, exist_ok=True)
np.save(OUT / "embeddings.npy", emb)
(OUT / "chunks.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
(OUT / "config.json").write_text(json.dumps({"model": MODEL, "count": len(records)}, indent=2), encoding="utf-8")
print(f"Built RAG index with {len(records)} chunks -> {OUT}")
