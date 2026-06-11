# 📄 DocChat — RAG Chatbot with Gemini

## 🚀 Live Demo
👉 [Click here to try the app](https://rag-chat-sudeep.streamlit.app/)

A production-style **Retrieval-Augmented Generation (RAG)** pipeline that lets you chat with any PDF document. Built with LangChain, ChromaDB, sentence-transformers, and Gemini 1.5 Flash.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-orange)

---

## 🚀 Features

- 📤 Upload any PDF — research papers, reports, notes
- 🔍 Semantic search using `all-MiniLM-L6-v2` embeddings
- 🧠 Answers grounded in document context (no hallucination)
- 📍 Page-level source citations on every answer
- 🔎 View retrieved chunks with similarity scores
- 💬 Multi-turn chat history in session

---

## 🏗️ Architecture

```
PDF Upload
    ↓
Text Extraction (PyMuPDF)
    ↓
Chunking with overlap (LangChain RecursiveCharacterTextSplitter)
    ↓
Embedding (sentence-transformers: all-MiniLM-L6-v2)
    ↓
Vector Store (NumPy cosine similarity, in-memory)
    ↓
User Question → Embed → Cosine similarity search → Top 5 chunks
    ↓
Prompt = chunks + question → Gemini 1.5 Flash
    ↓
Answer + Page citations
```

---

## 📁 Project Structure

```
rag-chatbot/
├── app.py                  ← Streamlit frontend
├── rag/
│   ├── loader.py           ← PDF text extraction (PyMuPDF)
│   ├── chunker.py          ← Text splitting with overlap
│   ├── embedder.py         ← Embedding + ChromaDB storage
│   ├── retriever.py        ← Semantic similarity search
│   └── generator.py        ← Gemini API call + answer
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get free Gemini API key
- Go to [aistudio.google.com](https://aistudio.google.com)
- Click **Get API Key** → **Create API key**
- Copy the key (starts with `AIza...`)

### 5. Run the app
```bash
streamlit run app.py
```

Enter your API key in the sidebar, upload a PDF, and start chatting!

---

## 🌐 Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set main file as `app.py`
4. Under **Settings → Secrets**, add:
```toml
GEMINI_API_KEY = "AIza_your_key_here"
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| PDF Parsing | PyMuPDF (fitz) |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | NumPy cosine similarity (in-memory) |
| LLM | Gemini 1.5 Flash (Google AI) |

---

## 👤 Author

**Sudeep** — [GitHub](https://github.com/Sudeepcpc)

CSE Graduate · AI/ML Engineer · Building real-world AI systems
