from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector database
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

# LLM
llm = ChatMistralAI(
    model="mistral-small-latest"
)

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that answers questions based only on the provided context. "
            "If the answer is not found in the context, say: "
            "'I could not find the answer in the documentation.'"
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ]
)

print("RAG system is ready.")
print("Press 0 to exit.")

while True:

    query = input("\nYou: ")

    if query == "0":
        break

    # Retrieve documents
    docs = retriever.invoke(query)

    # Convert docs into context
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Create prompt
    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    # Generate response
    response = llm.invoke(final_prompt)

    print("\nAssistant:", response.content)
        