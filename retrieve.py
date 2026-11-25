from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from config import BOOK_ID, CHROMA_DIR, COLLECTION, EMBED_MODEL

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _load():
    try:
        model = SentenceTransformer(EMBED_MODEL, local_files_only=True)
    except Exception:
        model = SentenceTransformer(EMBED_MODEL)
    col = chromadb.PersistentClient(CHROMA_DIR).get_collection(COLLECTION)
    return model, col


def retrieve(query, k=5, k_raw=12):
    """Return (top-k hits with scores, neighbouring chunks of the best hits).

    Strategy:
    1. Fetch k_raw candidates from ChromaDB.
    2. Drop exercise chunks (end-of-chapter Q&A) — they look semantically
       similar to any question but contain no explanatory content.
    3. Prefer concept chunks; fall back to activity chunks only if fewer
       than k concept chunks are available.
    4. Expand context by fetching idx±1 neighbours for the top-2 hits
       (not just top-1), so multi-part explanations are covered.
    """
    model, col = _load()
    q = model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()
    r = col.query(
        query_embeddings=[q],
        n_results=k_raw,
        include=["documents", "metadatas", "distances"],
    )

    all_hits = [
        {"text": d, "meta": m, "score": 1 - dist}
        for d, m, dist in zip(
            r["documents"][0], r["metadatas"][0], r["distances"][0]
        )
    ]

    # P0 Fix: drop exercise chunks entirely
    all_hits = [h for h in all_hits if h["meta"]["chunk_type"] != "exercise"]

    # P0 Fix: prefer concept chunks; only use activities as fallback.
    # Within each group, preserve the original score ranking.
    concept_hits = [h for h in all_hits if h["meta"]["chunk_type"] == "concept"]
    activity_hits = [h for h in all_hits if h["meta"]["chunk_type"] == "activity"]

    # If the highest-scoring activity beats the best concept by >0.05, include
    # it in top position so genuinely relevant activity context is not lost.
    if (
        activity_hits
        and concept_hits
        and (activity_hits[0]["score"] - concept_hits[0]["score"]) > 0.05
    ):
        hits = ([activity_hits[0]] + concept_hits)[:k]
    else:
        hits = (concept_hits + activity_hits)[:k]

    if not hits:
        return [], []

    # P1 Fix: fetch neighbours for the top-2 hits (not just top-1)
    neighbour_id_to_chapter: dict[str, int] = {}
    for top_hit in hits[:2]:
        b = top_hit["meta"]
        for j in (b["idx"] - 1, b["idx"] + 1):
            if j >= 0:
                nid = f"{BOOK_ID}-{j:05d}"
                neighbour_id_to_chapter[nid] = b["chapter_no"]

    neighbours: list[dict] = []
    if neighbour_id_to_chapter:
        n = col.get(
            ids=list(neighbour_id_to_chapter.keys()),
            include=["documents", "metadatas"],
        )
        neighbours = [
            {"text": d, "meta": m}
            for d, m in zip(n["documents"], n["metadatas"])
            if m["chapter_no"] == neighbour_id_to_chapter.get(
                f"{BOOK_ID}-{m['idx']:05d}"
            )
            and m["chunk_type"] != "exercise"
        ]
        neighbours.sort(key=lambda c: c["meta"]["idx"])

    return hits, neighbours
