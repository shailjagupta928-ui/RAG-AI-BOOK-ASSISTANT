AI Book Assistant 📚🤖

A Retrieval-Augmented Generation (RAG) based AI application that allows users to upload PDF books or documents and ask questions about their content. The system retrieves the most relevant information from the uploaded document and uses a Large Language Model (LLM) to generate accurate, context-aware responses.

🚀 Features
📄 Upload and process PDF documents
✂️ Split documents into smaller chunks for efficient retrieval
🧠 Generate semantic embeddings using Hugging Face
🔎 Perform similarity-based document retrieval
🗄️ Store and search embeddings using ChromaDB
🤖 Generate answers using Mistral LLM
💬 Interactive question-answering interface
📚 Answers grounded in the uploaded document
⚡ Fast and efficient retrieval pipeline
🛡️ Reduces hallucinations by providing relevant document context

                ┌─────────────────┐
                │    PDF Upload   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  PDF Text       │
                │  Extraction     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Text Chunking   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   Embeddings    │
                │ Hugging Face    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │    ChromaDB     │
                │ Vector Database  │
                └────────┬────────┘
                         │
              User Question
                         │
                         ▼
                ┌─────────────────┐
                │ Semantic Search │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Relevant Context│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   Mistral LLM   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │     Answer      
                └────────────────

🛠️ Tech Stack
Technology	   Purpose
Python	        Core development
LangChain	      RAG pipeline orchestration
Mistral AI	     Large Language Model
Hugging Face	   Text embeddings
ChromaDB	       Vector database
PyPDFLoader	     PDF document loading
Streamlit	       Web interface

AI-BOOK-ASSISTANT/
│
├── app.py
├── requirements.txt
├── .env
├── README.md
│
├── data/
│   └── sample_book.pdf
│
└── src/
    ├── document_loader.py
    ├── embeddings.py
    ├── vector_store.py
    └── rag_pipeline.py
