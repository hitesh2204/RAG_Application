import os
import streamlit as st
from PIL import Image
import pytesseract
from langchain_community.document_loaders import TextLoader, PyPDFLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceHub
from langchain_core.runnables import RunnableSequence
from langchain_core.documents import Document
#from multimodel_rag import load_all_data,chunk_documents,build_vector_store,get_llm,get_chain
# Set Tesseract path

# Title
st.title("📚 Multimodal RAG App By Hitesh(PDF + YouTube + Image + Text)")

# Inputs
user_question = st.text_input("🔍 Enter your query")
youtube_url = st.text_input("📺 Enter YouTube video URL")

# File Uploads
text_file = st.file_uploader("📄 Upload a Text file", type=["txt"])
pdf_file = st.file_uploader("📕 Upload a PDF", type=["pdf"])
image_file = st.file_uploader("🖼️ Upload an Image", type=["jpg", "jpeg", "png"])


# === Helper Functions ===

def load_all_data(text_file, pdf_file, image_file, youtube_url):
    docs = []

    # Text
    if text_file:
        path = f"temp_{text_file.name}"
        with open(path, "wb") as f:
            f.write(text_file.read())
        docs += TextLoader(path, encoding="utf-8").load()

    # PDF
    if pdf_file:
        path = f"temp_{pdf_file.name}"
        with open(path, "wb") as f:
            f.write(pdf_file.read())
        docs += PyPDFLoader(path).load()

    # Image OCR
    if image_file:
        image = Image.open(image_file)
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        image_text = pytesseract.image_to_string(image)
        docs += [Document(page_content=image_text)]

    # YouTube
    if youtube_url:
        try:
            yt_docs = YoutubeLoader.from_youtube_url(youtube_url).load()
            docs += yt_docs
        except Exception as e:
            st.error(f"YouTube error: {str(e)}")

    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def build_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embedding=embeddings)

def get_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.3, "max_new_tokens": 512}
    )

def get_chain(llm, retriever):
    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant. Based on the following documents:\n\n{context}\n\n"
            "Answer the question: {question}\n\n"
            "If unsure, say 'I don't know'."
        ),
        input_variables=["context", "question"]
    )
    return RunnableSequence(
        retriever | (lambda x: {
            "context": "\n\n".join([doc.page_content for doc in x]),
            "question": x[0]["question"]
        }) | prompt | llm | StrOutputParser()
    )


# === Run the App ===
if st.button("🧠 Generate Answer"):
    if not user_question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing..."):
            docs = load_all_data(text_file, pdf_file, image_file, youtube_url)
            if not docs:
                st.error("Please upload at least one document or enter a valid YouTube URL.")
            else:
                chunks = chunk_documents(docs)
                vector_store = build_vector_store(chunks)
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                llm = get_llm()
                chain = get_chain(llm, retriever)

                result = chain.invoke({"question": user_question})
                st.success("✅ Answer:")
                st.write(result)
