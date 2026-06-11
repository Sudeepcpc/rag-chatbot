import chromadb
from sentence_transformers import SentenceTransformer

# Load model once at module level (cached after first load)
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def store_chunks(chunks: list[dict], collection_name: str = "rag_docs") -> chromadb.Collection:
    """
    Embed chunks using all-MiniLM-L6-v2 and store in ChromaDB (in-memory).
    Returns the ChromaDB collection for retrieval.
    """
    model = get_model()
    client = chromadb.Client()  # in-memory, no disk needed

    # Delete existing collection if it exists (for re-uploads)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(collection_name)

    texts = [c["text"] for c in chunks]
    metadatas = [{"page": c["page"]} for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    # Embed in batches to avoid memory issues
    batch_size = 64
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False).tolist()
        all_embeddings.extend(embeddings)

    collection.add(
        documents=texts,
        embeddings=all_embeddings,
        ids=ids,
        metadatas=metadatas
    )

    return collection
