import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")

INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(DATA_DIR, "policy_chunks.json")

TOP_K = 5
SIMILARITY_THRESHOLD = 0.50
MIN_SCORE = 0.55

# ---------- EMBEDDING ----------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

index = None
policy_chunks = []
_last_index_mtime = None
_last_chunks_mtime = None


def _choose_chunks_path() -> str:
    """Use the active policy chunks generated from demo PDFs or uploaded PDFs."""
    return CHUNKS_PATH


def _mtime(path: str):
    return os.path.getmtime(path) if os.path.exists(path) else None


def reload_retriever_data(force: bool = False) -> None:
    """Reload FAISS/chunks when source files changed on disk."""
    global index, policy_chunks, _last_index_mtime, _last_chunks_mtime

    chunks_path = _choose_chunks_path()
    index_mtime = _mtime(INDEX_PATH)
    chunks_mtime = _mtime(chunks_path)

    if not force and index_mtime == _last_index_mtime and chunks_mtime == _last_chunks_mtime:
        return

    try:
        if index_mtime is None or chunks_mtime is None:
            index = None
            policy_chunks = []
            _last_index_mtime = index_mtime
            _last_chunks_mtime = chunks_mtime
            return

        index = faiss.read_index(INDEX_PATH)
        with open(chunks_path, "r", encoding="utf-8") as f:
            policy_chunks = json.load(f)

        if index.ntotal != len(policy_chunks):
            print(f"[WARN]: FAISS mismatch: index={index.ntotal}, chunks={len(policy_chunks)}")

        _last_index_mtime = index_mtime
        _last_chunks_mtime = chunks_mtime
    except Exception as e:
        print(f"[ERROR]: Retriever failed to load: {e}")
        index = None
        policy_chunks = []


# Initial load at import time.
reload_retriever_data(force=True)


def retrieve_policy_chunks(query: str):
    """
    Retrieve relevant policy chunks using FAISS.
    Enforces Strict semantic re-ranking and consistency filtering.
    """
    reload_retriever_data()

    if not index or not policy_chunks:
        print("[WARN]: Retriever not initialized or empty.")
        return []

    print("Query:", query)

    # Encode query
    query_emb = embedder.encode([query], normalize_embeddings=True)
    query_emb = np.asarray(query_emb).astype("float32")

    # Search FAISS
    scores, indices = index.search(query_emb, TOP_K)

    results = []
    
    # Track unique sections for consistency filter
    sections_seen = set()

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        
        chunk = policy_chunks[idx]
        
        # TASK 1: Add semantic re-ranking
        chunk_text = chunk.get("text", "")
        chunk_emb = embedder.encode([chunk_text], normalize_embeddings=True)
        chunk_emb = np.asarray(chunk_emb).astype("float32")
        
        # Cosine similarity dot product
        semantics_score = float(np.dot(query_emb[0], chunk_emb[0]))
        
        if semantics_score >= MIN_SCORE:
            section = chunk.get("section_title", "General")
            
            res = {
                "text": chunk_text,
                "metadata": {
                    "policy_id": chunk.get("policy_id", "DEMO_POLICY"),
                    "section": section,
                    "clause_id": chunk.get("clause_id", str(idx))
                },
                "score": semantics_score
            }
            results.append(res)
            sections_seen.add(section)

    # LIMIT results to top 3 and Sort
    results.sort(key=lambda x: x["score"], reverse=True)
    results = results[:3]
    
    # TASK 1: Add consistency filter
    unique_sections = set(res["metadata"]["section"] for res in results)
    if len(unique_sections) > 2 and results:
        highest_scoring_section = results[0]["metadata"]["section"]
        results = [res for res in results if res["metadata"]["section"] == highest_scoring_section]

    print("Retrieved:", len(results))
    print("Top clauses:", [c["metadata"]["clause_id"] for c in results])
    
    return results
