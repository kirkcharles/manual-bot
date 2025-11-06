import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os

# --- CONFIG ---
MANUAL_PATH = "manual.pdf"

# --- Load & Index (cached) ---
@st.cache_resource
def get_qa_chain():
    if not os.path.exists(MANUAL_PATH):
        st.error(f"PDF not found: {MANUAL_PATH}")
        st.stop()

    loader = PyPDFLoader(MANUAL_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-small"))
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(k=4)
    )
    return qa

qa = get_qa_chain()

# --- UI ---
st.title("Company Manual Bot")
st.caption("Ask questions – answers come straight from your user manual")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("e.g. How do I reset the printer?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching manual..."):
            result = qa.invoke(prompt)
            answer = result["result"]
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})