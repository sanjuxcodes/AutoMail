import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[1]
RAG_DATA_DIR = BASE_DIR / "rag" / "data"
DATA_PATH = RAG_DATA_DIR / "policy_chunks.json"
INDEX_PATH = RAG_DATA_DIR / "faiss.index"


def main() -> None:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c.get("text", "") for c in chunks if c.get("text")]
    if not texts:
        raise ValueError(f"No usable chunks found in {DATA_PATH}")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    print(f"FAISS index rebuilt with {len(texts)} policy chunks at {INDEX_PATH}.")


if __name__ == "__main__":
    main()
