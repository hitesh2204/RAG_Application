from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()

### 1. Load and Split PDF
def load_and_split_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)
    return docs

### 2. Create Embeddings and Vector Store
def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embedding=embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever

### 3. LLM for text generation (compatible model)
def get_llm():
    llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-xl",
    task="text-generation")
    return llm

def hf_model():
    llm=get_llm()
    return ChatHuggingFace(llm=llm)


### 4. Prompt Template
prompt = PromptTemplate(
    template=(
        "You are a helpful AI assistant. Based on the following documents:\n\n{documents}\n\n"
        "Answer the user's question: {user_query}\n\n"
        "If you don't know the answer, say 'I don't know'."
    ),
    input_variables=["documents", "user_query"]
)

### 5. Main Execution
def main():
    pdf_path = "D://RAG_Application//PDF Q&A Bot//Data//ML cheetsheet.pdf"
    docs = load_and_split_pdf(pdf_path)
    retriever = create_vector_store(docs)
    
    user_query = "What is linear regression?"
    relevant_docs = retriever.invoke(user_query)

    # Combine document contents into a single string
    documents_text = "\n\n".join([doc.page_content for doc in relevant_docs])

    model = hf_model()

    chain = RunnableSequence(prompt | model | StrOutputParser())
    response = chain.invoke({
        "documents": documents_text,
        "user_query": user_query
    })

    print("\n💬 Answer:")
    print(response)

if __name__ == "__main__":
    main()
