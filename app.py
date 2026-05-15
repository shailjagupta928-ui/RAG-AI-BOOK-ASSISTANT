import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import tempfile
import os

load_dotenv()

st.set_page_config(
    page_title="📚 Book Assistant",
    page_icon="🤖"
)

st.title("📚 AI Book Assistant 🤖")
st.write("Upload any PDF and chat with it ✨")

# Upload PDF
uploaded_file = st.file_uploader(
    "📄 Upload PDF",
    type="pdf"
)

if uploaded_file:

    # Save temp pdf
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("✅ PDF Uploaded")

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create Vector DB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    retriever = vectorstore.as_retriever()

    st.success("📦 Vector Database Created")

    # LLM
    llm = ChatMistralAI(
        model="mistralai/Mistral-Large-Instruct-2407"
    )

    # Chat input
    query = st.chat_input("💬 Ask question from book")

    if query:

        with st.spinner("🤔 Thinking..."):

            docs = retriever.invoke(query)

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            prompt = f"""
            Answer only from the given context.

            Context:
            {context}

            Question:
            {query}
            """

            response = llm.invoke(prompt)

            st.chat_message("user").write(query)
            st.chat_message("assistant").write(response.content)