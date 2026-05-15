from langchain_community.document_loaders import TextLoader

data = TextLoader(r"C:\Users\HP\Desktop\RAG Project\document_loaders\notes.txt")
docs = data.load()
print(docs)
