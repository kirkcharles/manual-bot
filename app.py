# app.py — FAISS version (no chromadb needed)
import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import streamlit as st

# Load PDFs
from langchain_community.document_loaders import PyPDFLoader

# Vector store: FAISS (installed: faiss-cpu)
from langchain_community.vectorstores import FAISS

# OpenAI wrappers
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Text splitter (new package name, with fallback to legacy path)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # legacy


# --------------------------- UI CONFIG --------------------------- #
st.set_page_config(page_title="Manual Bot", page_icon="📚", layout="wide")
st.title("📚 Manual Bot")
st.caption("Upload PDFs or use PDFs in your repo (e.g., manual.pdf) → index with FAISS → ask questions. OpenAI embeddings + ChatOpenAI (LangChain 1.x).")

# --------------------------- ENV / KEYS --------------------------- #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.warning("⚠️ Set the environment variable `OPENAI_API_KEY` for the bot to respond.")

# --------------------------- SIDEBAR --------------------------- #
with st.sidebar:
    st.header("Settings")
    llm_model = st.selectbox(
        "Chat model",
        options=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        index=0,
    )
    embed_model = st.selectbox(
        "Embedding model",
        options=["text-embedding-3-small", "text-embedding-3-large"],
        index=0,
    )
    chunk_size = st.slider("Chunk size", 300, 3000, 1000, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, 10)
    k_docs = st.slider("Top-K documents", 1, 10, 4, 1)

    st.divider()
    auto_index_repo = st.checkbox("Auto-index repo PDFs on first run (if no index yet)", value=True)
    st.caption("Re-index after changing chunking. Index is persisted to a temp dir for this session.")

# --------------------------- PERSISTENCE --------------------------- #
# Use a temp folder for FAISS persistence
if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = str(Path(tempfile.gettempdir()) / "manual_bot_faiss")
PERSIST_DIR = st.session_state.persist_dir

# Repo root (where this file lives)
REPO_DIR = Path(__file__).resolve().parent


# --------------------------- HELPERS --------------------------- #
def load_pdfs_to_docs(uploaded_files) -> List:
    """Load uploaded PDFs into LangChain Documents via PyPDFLoader."""
    docs = []
    for up in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(up.read())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            file_docs = loader.load()
            for d in file_docs:
                d.metadata = d.metadata or {}
                d.metadata["source"] = up.name
            docs.extend(file_docs)
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
    return docs


def load_repo_pdfs_to_docs(paths: List[Path]) -> List:
    """Load repo PDFs (existing on disk) into LangChain Documents."""
    docs = []
    for p in paths:
        try:
            loader = PyPDFLoader(str(p))
            file_docs = loader.load()
            for d in file_docs:
                d.metadata = d.metadata or {}
                d.metadata["source"] = p.name
            docs.extend(file_docs)
        except Exception as e:
            st.error(f"Failed to load {p.name}: {e}")
    return docs


def find_repo_pdfs(repo_dir: Path) -> List[Path]:
    """Discover PDFs in the repo root, prioritising manual.pdf."""
    candidates: List[Path] = []
    manual = repo_dir / "manual.pdf"
    if manual.exists() and manual.is_file():
        candidates.append(manual)
    for p in sorted(repo_dir.glob("*.pdf")):
        if p not in candidates:
            candidates.append(p)
    return candidates


def clear_index(persist_dir: str):
    """Delete the FAISS persist directory to rebuild from scratch."""
    root = Path(persist_dir)
    if not root.exists():
        return
    for p in root.glob("**/*"):
        try:
            if p.is_file():
                p.unlink(missing_ok=True)
        except Exception:
            pass
    for p in sorted(root.glob("**/*"), reverse=True):
        try:
            if p.is_dir():
                p.rmdir()
        except Exception:
            pass
    try:
        root.rmdir()
    except Exception:
        pass


def save_faiss_index(vs: FAISS, path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
    vs.save_local(path)


def load_faiss_index(path: str, embeddings: OpenAIEmbeddings) -> FAISS | None:
    p = Path(path)
    if not p.exists():
        return None
    # allow_dangerous_deserialization is required by FAISS loader in LangChain
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def build_or_load_vectorstore(
    docs: List,
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    persist_dir: str,
) -> FAISS:
    """
    Create or load FAISS vectorstore.
    If docs provided, (re)build index and save; otherwise try loading from disk.
    """
    embeddings = OpenAIEmbeddings(model=embedding_model_name, api_key=OPENAI_API_KEY)

    if docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        vs = FAISS.from_documents(chunks, embeddings)
        save_faiss_index(vs, persist_dir)
        return vs

    loaded = load_faiss_index(persist_dir, embeddings)
    if loaded is None:
        # Create an empty index so app logic can proceed
        vs = FAISS.from_texts([""], embeddings, metadatas=[{"source": "empty"}])
        save_faiss_index(vs, persist_dir)
        return vs
    return loaded


def simple_rag_answer(vs: FAISS, model_name: str, query: str, k: int) -> Tuple[str, List]:
    """Minimal RAG: retrieve top-k, build prompt, call LLM."""
    retriever = vs.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)

    def snippet(t: str, limit: int = 1200) -> str:
        t = t.strip().replace("\n", " ").replace("  ", " ")
        return t[:limit] + ("…" if len(t) > limit else "")

    context_blocks = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source", "unknown.pdf")
        page = meta.get("page", "?")
        context_blocks.append(f"[{i}] {src} (page {page})\n{snippet(d.page_content)}")

    context = "\n\n".join(context_blocks) if context_blocks else "No context retrieved."

    system_preamble = (
        "You are a helpful technical assistant. Answer using the provided context. "
        "If the answer is not in the context, say you don't know."
    )
    user_prompt = (
        f"{system_preamble}\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer concisely and cite sources by their bracket number like [1], [2]."
    )

    llm = ChatOpenAI(model=model_name, temperature=0.2, api_key=OPENAI_API_KEY)
    resp = llm.invoke(user_prompt)
    answer_text = getattr(resp, "content", "") if resp else ""
    return answer_text.strip(), docs


# --------------------------- MAIN UI --------------------------- #
st.subheader("1) Index PDFs")

# Discover repo PDFs (e.g., manual.pdf)
repo_pdf_paths = find_repo_pdfs(REPO_DIR)
if repo_pdf_paths:
    st.success(f"Found {len(repo_pdf_paths)} PDF(s) in repo: " + ", ".join(p.name for p in repo_pdf_paths))
else:
    st.info("No PDFs found in the repo directory. (Looking in the same folder as app.py)")

uploaded = st.file_uploader("Drag & drop PDFs to add", type=["pdf"], accept_multiple_files=True)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    rebuild_uploaded = st.button("🔨 Re-index uploaded PDFs", type="primary", disabled=(not uploaded))
with col2:
    rebuild_repo = st.button("📦 Index repo PDFs (e.g., manual.pdf)", type="secondary", disabled=(not repo_pdf_paths))
with col3:
    wiped = st.button("🗑️ Clear existing index", type="secondary")

if wiped:
    clear_index(PERSIST_DIR)
    st.success("Cleared index. Rebuild with new PDFs when ready.")

vs = None
indexed_from = None

# Option 1: Index uploaded PDFs
if rebuild_uploaded and uploaded:
    with st.spinner("Building index from uploaded PDFs…"):
        docs = load_pdfs_to_docs(uploaded)
        vs = build_or_load_vectorstore(
            docs=docs,
            embedding_model_name=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            persist_dir=PERSIST_DIR,
        )
        indexed_from = f"{len(uploaded)} uploaded file(s)"
        st.success(f"Indexed {len(docs)} pages from {indexed_from}.")

# Option 2: Index repo PDFs
if rebuild_repo and repo_pdf_paths:
    with st.spinner("Building index from repo PDFs…"):
        docs = load_repo_pdfs_to_docs(repo_pdf_paths)
        vs = build_or_load_vectorstore(
            docs=docs,
            embedding_model_name=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            persist_dir=PERSIST_DIR,
        )
        indexed_from = f"{len(repo_pdf_paths)} repo file(s)"
        st.success(f"Indexed {len(docs)} pages from {indexed_from}.")

# If no explicit rebuild, try to load existing store (and optionally auto-index repo PDFs)
if vs is None:
    index_exists = Path(PERSIST_DIR).exists() and any(Path(PERSIST_DIR).glob("**/*"))
    try:
        if not index_exists and auto_index_repo and repo_pdf_paths:
            with st.spinner("No existing index. Auto-indexing repo PDFs…"):
                docs = load_repo_pdfs_to_docs(repo_pdf_paths)
                vs = build_or_load_vectorstore(
                    docs=docs,
                    embedding_model_name=embed_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    persist_dir=PERSIST_DIR,
                )
                indexed_from = f"{len(repo_pdf_paths)} repo file(s) (auto)"
                st.success(f"Indexed {len(docs)} pages from {indexed_from}.")
        else:
            # Load existing FAISS index (or create empty)
            vs = build_or_load_vectorstore(
                docs=[],
                embedding_model_name=embed_model,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                persist_dir=PERSIST_DIR,
            )
    except Exception as e:
        st.info("No existing index yet. Upload PDFs or click **Index repo PDFs**.")
        st.caption(str(e))

st.divider()
st.subheader("2) Ask questions about your manuals")

query = st.text_input("Your question", placeholder="e.g., What does the safety section say about battery isolation?")
ask = st.button("💬 Ask")

if ask:
    if not OPENAI_API_KEY:
        st.error("Missing `OPENAI_API_KEY`. Set it in your environment.")
    elif vs is None:
        st.error("No vector store available. Upload PDFs or index repo PDFs first.")
    else:
        try:
            with st.spinner("Thinking…"):
                answer, sources = simple_rag_answer(vs, llm_model, query, k_docs)

            st.markdown("### ✅ Answer")
            st.write(answer or "_No answer produced._")

            if sources:
                st.markdown("### 📎 Sources")
                for i, doc in enumerate(sources, start=1):
                    meta = doc.metadata or {}
                    src = meta.get("source", "unknown.pdf")
                    page = meta.get("page", "?")
                    st.markdown(f"- **[{i}] {src}** (page {page})")
        except Exception as e:
            st.exception(e)

st.divider()
with st.expander("ℹ️ Debug / Info"):
    st.write("Repo dir:", str(REPO_DIR))
    st.write("Repo PDFs found:", [p.name for p in repo_pdf_paths])
    for p in repo_pdf_paths:
        try:
            st.write(f"- {p.name}: exists={p.exists()}, size={p.stat().st_size} bytes")
        except Exception:
            st.write(f"- {p.name}: (stat unavailable)")
    st.write("Persist directory:", PERSIST_DIR)
    st.write("LLM model:", llm_model)
    st.write("Embedding model:", embed_model)
    st.write("Chunking:", {"size": chunk_size, "overlap": chunk_overlap})
    st.write("Top-K:", k_docs)
