#load pdf
#split into chunks
#create embeddings
#store into chroma

import langchain_community
from langchain_core import documents
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

loader = PyPDFLoader(r"C:\Users\HP\Desktop\RAG Project\document_loaders\deeplearning.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
persist_directory = "chroma_db"
vectorstore = Chroma.from_documents(documents=chunks, embedding=embedding, persist_directory=persist_directory)