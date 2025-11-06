# app.py
import os
import tempfile
from pathlib import Path
from typing import List

import streamlit as st

# LangChain community integrations
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

# LangChain core/LLM + embeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Fallback-safe import for the splitter (works with old & new langchain layouts)
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:  # legacy path, for older langchain versions
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

from langchain.chains import RetrievalQA


# --------------------------- UI CONFIG --------------------------- #
st.set_page_config(
    page_title="Manual Bot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Manual Bot")
st.caption(
    "Upload PDFs, index them locally with Chroma, and ask questions. "
    "Uses OpenAI embeddings + ChatOpenAI via LangChain 1.x."
)

# --------------------------- ENV / KEYS --------------------------- #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.warning(
        "⚠️ The environment variable `OPENAI_API_KEY` is not set. "
        "Set it in your deployment settings for the app to respond."
    )

# --------------------------- SIDEBAR --------------------------- #
with st.sidebar:
    st.header("Settings")
    llm_model = st.selectbox(
        "Chat model",
        # Pick a light, fast default; adjust if you prefer
        options=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        index=0,
    )
    embed_model = st.selectbox(
        "Embedding model",
        options=["text-embedding-3-small", "text-embedding-3-large"],
        index=0,
    )
    chunk_size = st.slider("Chunk size", min_value=300, max_value=3000, value=1000, step=50)
    chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=400, value=150, step=10)

    st.divider()
    st.caption(
        "Tip: re-index after changing chunking/collection. "
        "Collections are persisted to a per-run temp dir."
    )

# --------------------------- PATHS / PERSISTENCE --------------------------- #
# Use a temp folder for Chroma persistence on Streamlit Cloud/Community
# (ephemeral across rebuilds, persistent across reruns in the same session)
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
        # Save to a temp file first so PyPDFLoader can read it from disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(up.read())
            tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        file_docs = loader.load()
        # Attach metadata filename for traceability
        for d in file_docs:
            d.metadata = d.metadata or {}
            d.metadata["source"] = up.name
        docs.extend(file_docs)
        # Clean up temp file
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
    """Create (or load) a Chroma vectorstore. If docs provided, (re)build index."""
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

    # If no docs (yet), attempt to load existing collection (if it exists)
    return Chroma(
        embedding_function=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )


def make_qa_chain(vs: Chroma, model_name: str):
    """Create a simple RetrievalQA chain using 'stuff' combine strategy."""
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model=model_name, temperature=0.2, api_key=OPENAI_API_KEY)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # classic simple approach
        return_source_documents=True,
    )
    return chain


def clear_index(persist_dir: str):
    """Delete the Chroma persist directory to rebuild from scratch."""
    try:
        if Path(persist_dir).exists():
            for p in Path(persist_dir).glob("**/*"):
                try:
                    if p.is_file():
                        p.unlink(missing_ok=True)
                except Exception:
                    pass
            # Remove directories (post-order)
            for p in sorted(Path(persist_dir).glob("**/*"), reverse=True):
                try:
                    if p.is_dir():
                        p.rmdir()
                except Exception:
                    pass
            Path(persist_dir).rmdir()
    except Exception:
        pass


# --------------------------- MAIN UI --------------------------- #
st.subheader("1) Upload PDFs to index")
uploaded = st.file_uploader(
    "Drag & drop one or more PDF manuals",
    type=["pdf"],
    accept_multiple_files=True,
)

col1, col2 = st.columns([1, 1])
with col1:
    rebuild = st.button("🔨 Re-index uploaded PDFs", type="primary", disabled=(not uploaded))
with col2:
    wiped = st.button("🗑️ Clear existing index", type="secondary")

# Handle index actions
if wiped:
    clear_index(PERSIST_DIR)
    # Bust cache_resource for the vectorstore
    build_or_load_vectorstore.clear()
    st.success("Cleared index. Rebuild with new PDFs when ready.")

vs = None

if rebuild and uploaded:
    with st.spinner("Building index…"):
        docs = load_pdfs_to_docs(uploaded)
        build_or_load_vectorstore.clear()  # ensure fresh cache
        vs = build_or_load_vectorstore(
            docs=docs,
            embedding_model_name=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            persist_dir=PERSIST_DIR,
            collection_name=COLLECTION,
        )
        st.success(f"Indexed {len(docs)} pages across {len(uploaded)} file(s).")

# If no rebuild, try to open existing store (if any)
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
    except Exception as e:
        st.info("No existing index found yet. Upload PDFs and click **Re-index**.")

st.divider()
st.subheader("2) Ask questions about your manuals")

query = st.text_input("Your question", placeholder="e.g., How do I reset the inverter?")
ask = st.button("💬 Ask")

if ask:
    if not OPENAI_API_KEY:
        st.error("Missing `OPENAI_API_KEY`. Set it in your environment to query the bot.")
    elif vs is None:
        st.error("No vector store available. Upload PDFs and click **Re-index** first.")
    else:
        try:
            chain = make_qa_chain(vs, llm_model)
            with st.spinner("Thinking…"):
                result = chain.invoke({"query": query})
            answer = result.get("result", "").strip()
            sources = result.get("source_documents", []) or []

            st.markdown("### ✅ Answer")
            st.write(answer if answer else "_No answer produced._")

            if sources:
                st.markdown("### 📎 Sources")
                for i, doc in enumerate(sources, start=1):
                    meta = doc.metadata or {}
                    src = meta.get("source", "unknown.pdf")
                    page = meta.get("page", "?")
                    st.markdown(f"- **{i}. {src}** (page {page})")
        except Exception as e:
            st.exception(e)

st.divider()
with st.expander("ℹ️ Debug / Info"):
    st.write("Persist directory:", PERSIST_DIR)
    st.write("Collection name:", COLLECTION)
    st.write("LLM model:", llm_model)
    st.write("Embedding model:", embed_model)
    st.write("Chunking:", {"size": chunk_size, "overlap": chunk_overlap})
