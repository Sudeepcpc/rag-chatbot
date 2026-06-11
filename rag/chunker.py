from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(pages: list[dict], chunk_size: int = 500, chunk_overlap: int = 60) -> list[dict]:
    """
    Split page texts into overlapping chunks.
    Overlap prevents losing context at chunk boundaries.
    Returns list of dicts: {text, page}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            if split.strip():
                chunks.append({
                    "text": split,
                    "page": page["page"]
                })
    return chunks
