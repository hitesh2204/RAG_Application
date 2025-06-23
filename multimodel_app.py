import streamlit as st
from PIL import Image
import pytesseract
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import HuggingFaceHub
from langchain_core.runnables import RunnableSequence

import os

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Helper functions
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents([text])

def build_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embedding=embeddings)

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
        docs = retriever.invoke(inputs)
        return {
            "context": "\n\n".join([doc.page_content for doc in docs]),
            "question": inputs["question"]
        }

    return RunnableSequence(
        prepare_inputs | prompt | llm | StrOutputParser()
    )

def get_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
    )

# Streamlit UI
st.title("🧠 Multimodal RAG Assistant")

text_file = st.file_uploader("Upload a text file", type=["txt"])
image_file = st.file_uploader("Upload an image with text", type=["png", "jpg", "jpeg"])
question = st.text_input("Ask a question based on your files")

if st.button("Get Answer"):
    if text_file and image_file and question:
        # Load files
        raw_text = text_file.read().decode("utf-8")
        image = Image.open(image_file)
        image_text = extract_text_from_image(image)

        combined_text = raw_text + "\n\n" + image_text
        docs = chunk_text(combined_text)
        vector_store = build_vector_store(docs)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        llm = get_llm()
        chain = get_chain(llm, retriever)
        response = chain.invoke({"question": question})

        st.subheader("📄 Answer")
        st.write(response)
    else:
        st.warning("Please upload both a text and image file, and enter a question.")
