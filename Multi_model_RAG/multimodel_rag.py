# 📦 Imports
# 📦 Imports
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

# 🧠 Step 1: Extract text from image
def extract_text_from_image(image_path):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)

# 📄 Step 2: Combine text file and image text
def combine_texts(text_file_path, image_path):
    with open(text_file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    image_text = extract_text_from_image(image_path)
    return raw_text + "\n\n" + image_text

# 🧱 Step 3: Split into chunks
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents([text])

# 🧠 Step 4: Embed and create vector store
def build_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embedding=embeddings)

# 🔗 Step 5: Prompt + Chain Setup
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

# 🧠 Step 6: Load LLM
def get_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
    )

# 🚀 Run Everything
def main():
    text_path = "D://RAG_Application//Multi_model_RAG//data//news.txt"
    image_path = "D://RAG_Application//Multi_model_RAG//data//rcb.jpeg"
    user_question = "Who won the 2025 IPL trophy?"

    print("📚 Loading and combining text & image...")
    combined_text = combine_texts(text_path, image_path)
    docs = chunk_text(combined_text)
    print("✅ Chunked into", len(docs), "documents")

    print("🔍 Creating vector store...")
    vector_store = build_vector_store(docs)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    print("🧠 Loading LLM...")
    llm = get_llm()
    chain = get_chain(llm, retriever)

    print("💬 Answering user query...")
    response = chain.invoke({"question": user_question})
    print("\n🧾 Final Answer:\n", response)

if __name__ == "__main__":
    main()
