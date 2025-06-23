# ✅ FULL Multimodal RAG App (PDF + YouTube + Image + Text) WITHOUT Streamlit

# 📦 Imports
import os
from PIL import Image
import pytesseract
import cv2
from langchain_community.document_loaders import TextLoader, PyPDFLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceHub
from langchain_core.runnables import RunnableSequence
from langchain_core.documents import Document



# 🧠 Step 1: Load Data from Multiple Sources
def load_all_data(text_path, pdf_path, image_path, youtube_url):
    # Text
    text_loader = TextLoader(text_path, encoding="utf-8")
    text_docs = text_loader.load()

    # PDF
    pdf_loader = PyPDFLoader(pdf_path)
    pdf_docs = pdf_loader.load()

    # Image OCR
    image = Image.open(image_path)
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    image_text = pytesseract.image_to_string(image)
    image_docs = [image_text]

    # YouTube Transcript
    yt_loader = YoutubeLoader.from_youtube_url(youtube_url)
    yt_docs = yt_loader.load()

    return text_docs + pdf_docs + [Document(page_content=image_text)] + yt_docs

# 🧱 Step 2: Chunking the Documents
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# 🔍 Step 3: Embedding & Vector DB
def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embedding=embeddings)

# 🧠 Step 4: LLM Setup
def get_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.3, "max_new_tokens": 512}
    )

# 🔗 Step 5: Create RAG Chain
def get_chain(llm, retriever):
    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant. Based on the following documents:\n\n{context}\n\n"
            "Answer the question: {question}\n\n"
            "If unsure, say 'I don't know'."
        ),
        input_variables=["context", "question"]
    )

    def prepare_inputs(inputs):
        docs = retriever.invoke(inputs["question"])
        return {
            "context": "\n\n".join([doc.page_content for doc in docs]),
            "question": inputs["question"]
        }

    return RunnableSequence(
        prepare_inputs | prompt | llm | StrOutputParser()
    )

# 🚀 Run Entire Flow
def main():
    text_path = "D://RAG_Application//Multimodel_rag_image_video_pdf_text//data//news.txt"
    pdf_path = "D://RAG_Application//Multimodel_rag_image_video_pdf_text//data//ML cheetsheet.pdf"
    image_path = "D://RAG_Application//Multimodel_rag_image_video_pdf_text//data//rcb.jpeg"
    youtube_url = "https://www.youtube.com/shorts/2dVG3B_OYaQ"  # Example URL
    user_query = "What did the video talk about?"

    print("🔄 Loading all data sources...")
    raw_docs = load_all_data(text_path, pdf_path, image_path, youtube_url)

    print("🧱 Splitting documents into chunks...")
    chunks = chunk_documents(raw_docs)

    print("🔍 Building vector store...")
    vector_db = build_vector_store(chunks)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    print("🧠 Loading LLM...")
    llm = get_llm()
    chain = get_chain(llm, retriever)

    print("💬 Answering query...")
    result = chain.invoke({"question": user_query})
    print("\n📜 Final Answer:\n", result)

if __name__ == "__main__":
    main()
