# app.py
import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import streamlit as st

# Community integrations
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

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
st.caption("Upload PDFs → index with Chroma → ask questions. OpenAI embeddings + ChatOpenAI (LangChain 1.x).")

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
    st.caption("Re-index after changing chunking. Index is persisted to a temp dir for this session.")

# --------------------------- PERSISTENCE --------------------------- #
if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = str(Path(tempfile.gettempdir()) / "manual_bot_chroma")
if "collection_name" not in st.session_state:
    st.session_state.collection_name = "manual-bot"

PERSIST_DIR = st.session_state.persist_dir
COLLECTION = st.session_state.collection_name

# --------------------------- HELPERS --------------------------- #
def load_pdfs_to_docs(uploaded_files) -> List:
    """Load each uploaded PDF into LangChain Documents via PyPDFLoader."""
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


@st.cache_resource(show_spinner=False)
def build_or_load_vectorstore(
    docs: List,
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    persist_dir: str,
    collection_name: str,
):
    """Create or load Chroma vectorstore. If docs provided, (re)build index."""
    embeddings = OpenAIEmbeddings(model=embedding_model_name, api_key=OPENAI_API_KEY)

    if docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)

        vs = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir,
        )
        vs.persist()
        return vs

    return Chroma(
        embedding_function=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )


def clear_index(persist_dir: str):
    """Delete the Chroma persist directory to rebuild from scratch."""
    root = Path(persist_dir)
    if not root.exists():
        return
    # remove files
    for p in root.glob("**/*"):
        try:
            if p.is_file():
                p.unlink(missing_ok=True)
        except Exception:
            pass
    # remove dirs (post-order)
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


def simple_rag_answer(
    vs: Chroma, model_name: str, query: str, k: int
) -> Tuple[str, List]:
    """
    Minimal RAG without langchain.chains:
    1) Retrieve top-k docs
    2) Build a compact prompt with context
    3) Call the LLM directly
    """
    retriever = vs.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)

    # assemble context (keep it under control to avoid token bloat)
    def snippet(t: str, limit: int = 1200) -> str:
        t = t.strip().replace("\n", " ").replace("  ", " ")
        return t[:limit] + ("…" if len(t) > limit else "")

    context_blocks = []
    for i, d in enumerate(docs, 1):
        src = (d.metadata or {}).get("source", "unknown.pdf")
        page = (d.metadata or {}).get("page", "?")
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
st.subheader("1) Upload PDFs to index")
uploaded = st.file_uploader("Drag & drop one or more PDF manuals", type=["pdf"], accept_multiple_files=True)

col1, col2 = st.columns([1, 1])
with col1:
    rebuild = st.button("🔨 Re-index uploaded PDFs", type="primary", disabled=(not uploaded))
with col2:
    wiped = st.button("🗑️ Clear existing index", type="secondary")

if wiped:
    clear_index(PERSIST_DIR)
    build_or_load_vectorstore.clear()
    st.success("Cleared index. Rebuild with new PDFs when ready.")

vs = None

if rebuild and uploaded:
    with st.spinner("Building index…"):
        docs = load_pdfs_to_docs(uploaded)
        build_or_load_vectorstore.clear()
        vs = build_or_load_vectorstore(
            docs=docs,
            embedding_model_name=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            persist_dir=PERSIST_DIR,
            collection_name=COLLECTION,
        )
        st.success(f"Indexed {len(docs)} pages across {len(uploaded)} file(s).")

# If not rebuilding now, try to load an existing store
if vs is None:
    try:
        vs = build_or_load_vectorstore(
            docs=[],
            embedding_model_name=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            persist_dir=PERSIST_DIR,
            collection_name=COLLECTION,
        )
    except Exception:
        st.info("No existing index yet. Upload PDFs and click **Re-index**.")

st.divider()
st.subheader("2) Ask questions about your manuals")

query = st.text_input("Your question", placeholder="e.g., How do I reset the inverter?")
ask = st.button("💬 Ask")

if ask:
    if not OPENAI_API_KEY:
        st.error("Missing `OPENAI_API_KEY`. Set it in your environment.")
    elif vs is None:
        st.error("No vector store available. Upload PDFs and click **Re-index** first.")
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
    st.write("Persist directory:", PERSIST_DIR)
    st.write("Collection name:", COLLECTION)
    st.write("LLM model:", llm_model)
    st.write("Embedding model:", embed_model)
    st.write("Chunking:", {"size": chunk_size, "overlap": chunk_overlap})
    st.write("Top-K:", k_docs)
