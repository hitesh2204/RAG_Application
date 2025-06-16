from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)
    return docs

def create_vector_store(docs):
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vector_store = FAISS.from_documents(docs, embedding=embeddings)
    vector_store.save_local("./faiss_db")
    return vector_store

# Load Hugging Face LLM
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)

def get_qa_chain(vector_store, llm):
    chat_llm = ChatHuggingFace(llm=llm)
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=chat_llm, retriever=retriever)
    return qa_chain

def main():
    docs = load_and_split_pdf("D://RAG_Application//PDF Q&A Bot//Data//ML cheetsheet.pdf")
    vector_store = create_vector_store(docs)
    qa_chain = get_qa_chain(vector_store, llm)
    user_query = "give me summary of linear regression algortihm?"
    response = qa_chain.run(user_query)
    print(response)

if __name__ == "__main__":
    main()
