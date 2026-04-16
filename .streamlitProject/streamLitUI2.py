import streamlit as st
from pdfminer.high_level import extract_text
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
import tempfile
import os
import httpx
import ssl

# ---------------------------
# Network / SSL setup
# ---------------------------
client = httpx.Client(verify=False)
ssl._create_default_https_context = ssl._create_unverified_context

os.environ["TIKTOKEN_CACHE_DIR"] = r"C:\\Users\\GenAIBLRANCUSR63\\Documents\\tittoken_cache"

# ---------------------------
# LLM + Embeddings
# ---------------------------
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",
    api_key="sk-f86P8es9uGdSMYl2aWt-Lw",
    http_client=client
)

embedding_model = OpenAIEmbeddings(
    base_url="https://genailab.tcs.in",
    model="azure/genailab-maas-text-embedding-3-large",
    api_key="sk-f86P8es9uGdSMYl2aWt-Lw",
    http_client=client
)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="RAG Chatbot")
st.title("💬 RAG PDF Chatbot")

# ---------------------------
# Session state
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# ---------------------------
# Upload PDF
# ---------------------------
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    with st.spinner("Processing PDF..."):
        raw_text = extract_text(temp_file_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_text(raw_text)

        vectordb = Chroma.from_texts(
            chunks,
            embedding_model,
            persist_directory="./chroma_index"
        )
        vectordb.persist()

        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )

        st.session_state.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever
        )

    st.sidebar.success("✅ PDF processed! You can now chat.")

    # Optional: auto summary once
    if not st.session_state.chat_history:
        with st.spinner("Generating summary..."):
            summary = st.session_state.rag_chain.invoke(
                "Give a concise summary of this document"
            )

        st.session_state.chat_history.append(("assistant", summary))

# ---------------------------
# Display chat
# ---------------------------
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# ---------------------------
# Chat input
# ---------------------------
query = st.chat_input("Ask something about the document...")

if query:
    st.session_state.chat_history.append(("user", query))

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ Thinking...")

        result = st.session_state.rag_chain.invoke(query)

        response = result if isinstance(result, str) else result.get("result", str(result))

        placeholder.empty()
        st.markdown(response)

    st.session_state.chat_history.append(("assistant", response))

# ---------------------------
# Clear chat
# ---------------------------
if st.sidebar.button("🗑 Clear Chat"):
    st.session_state.chat_history = []