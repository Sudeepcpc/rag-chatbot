import fitz  # PyMuPDF


def load_pdf(file) -> list[dict]:
    """
    Extract text from each page of a PDF.
    Returns list of dicts: {text, page}
    """
    doc = fitz.open(stream=file.read(), filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"text": text, "page": i + 1})
    return pages
