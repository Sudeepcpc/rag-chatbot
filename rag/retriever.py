from rag.embedder import get_model


def retrieve(query: str, collection, top_k: int = 5) -> list[dict]:
    """
    Embed the query and find top_k most similar chunks from ChromaDB.
    Returns list of dicts: {text, page, score}
    """
    model = get_model()
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for text, meta, dist in zip(docs, metas, distances):
        chunks.append({
            "text": text,
            "page": meta["page"],
            "score": round(1 - dist, 4)   # convert distance → similarity
        })

    return chunks
