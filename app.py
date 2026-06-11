import streamlit as st
from rag.loader import load_pdf
from rag.chunker import chunk_pages
from rag.embedder import store_chunks
from rag.retriever import retrieve
from rag.generator import generate_answer

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat — RAG with Gemini",
    page_icon="📄",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 800px; }
    .stChatMessage { border-radius: 12px; }
    .source-pill {
        display: inline-block;
        background: #E6F1FB;
        color: #185FA5;
        border: 1px solid #85B7EB;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 12px;
        margin: 2px;
    }
    .chunk-card {
        background: #F8F9FA;
        border-left: 3px solid #185FA5;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 8px;
        font-size: 13px;
        color: #444;
    }
    .stat-box {
        background: #F0F4FF;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────
st.title("📄 DocChat")
st.caption("RAG pipeline · Gemini 2.5 Flash · ChromaDB · sentence-transformers")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Setup")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Free key from aistudio.google.com → Get API Key"
    )

    st.divider()
    st.header("📁 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type="pdf",
        help="Upload any PDF — research paper, report, notes, etc."
    )

    if uploaded_file and api_key:
        if st.button("🔄 Process PDF", use_container_width=True):
            with st.spinner("Extracting text..."):
                pages = load_pdf(uploaded_file)

            with st.spinner(f"Chunking {len(pages)} pages..."):
                chunks = chunk_pages(pages)

            with st.spinner("Embedding & indexing into ChromaDB..."):
                collection = store_chunks(chunks)

            st.session_state["collection"] = collection
            st.session_state["doc_name"] = uploaded_file.name
            st.session_state["num_pages"] = len(pages)
            st.session_state["num_chunks"] = len(chunks)
            st.session_state["messages"] = []  # reset chat on new doc

            st.success("✅ Ready!")

    # Stats panel
    if "collection" in st.session_state:
        st.divider()
        st.markdown("**📊 Index Stats**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pages", st.session_state["num_pages"])
        with col2:
            st.metric("Chunks", st.session_state["num_chunks"])
        st.caption(f"📄 {st.session_state['doc_name']}")

    st.divider()
    st.caption("Built with LangChain · ChromaDB · Gemini · Streamlit")

# ── Main chat area ────────────────────────────────────────────

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Welcome message
if "collection" not in st.session_state:
    st.info("👈 Enter your Gemini API key and upload a PDF in the sidebar to get started.")
    st.stop()

# Render existing chat messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            source_html = " ".join(
                [f'<span class="source-pill">📍 Page {p}</span>'
                 for p in msg["sources"]]
            )
            st.markdown(source_html, unsafe_allow_html=True)
        if msg.get("chunks"):
            with st.expander("🔍 View retrieved chunks"):
                for c in msg["chunks"]:
                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<strong>Page {c["page"]}</strong> '
                        f'(score: {c["score"]})<br>{c["text"][:300]}...'
                        f'</div>',
                        unsafe_allow_html=True
                    )

# Chat input
query = st.chat_input("Ask anything about your document...")

if query:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
        st.stop()

    # Show user message
    with st.chat_message("user"):
        st.write(query)
    st.session_state["messages"].append({"role": "user", "content": query})

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching document..."):
            top_chunks = retrieve(query, st.session_state["collection"])

        if not top_chunks:
            answer = "I couldn't find relevant content for that question in the document."
            st.write(answer)
            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer
            })
        else:
            with st.spinner("Generating answer with Gemini..."):
                result = generate_answer(query, top_chunks, api_key)

            st.write(result["answer"])

            # Source citation pills
            source_html = " ".join(
                [f'<span class="source-pill">📍 Page {p}</span>'
                 for p in result["sources"]]
            )
            st.markdown(source_html, unsafe_allow_html=True)

            # Retrieved chunks expander
            with st.expander("🔍 View retrieved chunks"):
                for c in result["chunks"]:
                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<strong>Page {c["page"]}</strong> '
                        f'(similarity: {c["score"]})<br>{c["text"][:300]}...'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            st.session_state["messages"].append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "chunks": result["chunks"]
            })
