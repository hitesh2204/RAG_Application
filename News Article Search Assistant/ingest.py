# Importing libraries
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv
import os

load_dotenv()

### 1. Load and Split Text File
def load_and_split(path):
    loader = TextLoader(path, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    return chunks

### 2. Create Vector Store
def create_vector_store(chunks):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_documents(chunks, embedding=embedding)
    return vector_store

### ✅ 3. Load Chat Model from HuggingFaceHub
def my_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-large",  # Free-tier supported
        model_kwargs={"temperature": 0.3, "max_length": 512}
    )

### 4. Build Retrieval QA Chain
def get_qa_chain(vector_store, chat_llm):
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=chat_llm, retriever=retriever)
    return qa_chain

### 5. Main
def main():
    docs = load_and_split("D://RAG_Application//News Article Search Assistant//Data//news.txt")
    vector_store = create_vector_store(docs)
    chat_model = my_llm()
    qa_chain = get_qa_chain(vector_store, chat_model)

    user_query = "Who won the 2025 IPL trophy?"

    # ✅ Use invoke instead of deprecated .run()
    response = qa_chain.invoke({"query": user_query})

    print("\n🗨️ Answer:")
    print(response)

if __name__ == "__main__":
    main()
