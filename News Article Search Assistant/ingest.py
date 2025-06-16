### importing libraries.

#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from functools import partial
from dotenv import load_dotenv
import os

load_dotenv()

def load_and_split(path):
    loader = TextLoader(path,encoding="utf-8")
    documents = loader.load()

    ### splitting doc intochunks.
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    return chunks

def create_vector_store(chunks):
      model_name = "sentence-transformers/all-MiniLM-L6-v2"
      embedding=HuggingFaceEmbeddings(model_name=model_name)
      vector_store = FAISS.from_documents(chunks, embedding=embedding)
      return vector_store

def my_llm():
    llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation")
    return llm

def get_qa_chain(vector_store,llm):
      chat_llm=ChatHuggingFace(llm=llm)
      retriever=vector_store.as_retriever()
      qa_chain = RetrievalQA.from_chain_type(llm=chat_llm, retriever=retriever)
      return qa_chain

def main():
    docs = docs = load_and_split("D://RAG_Application//News Article Search Assistant//Data//news.txt") 
    vector_store = create_vector_store(docs)
    llm=my_llm()
    qa_chain = get_qa_chain(vector_store, llm)
    user_query = "who won the 2025 ipl tropy?"
    response = qa_chain.run(user_query)
    print(response)


if __name__ == "__main__":
    main()
