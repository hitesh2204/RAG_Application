# app.py
import streamlit as st
from chatbot import load_and_split_pdf, create_vector_store, get_qa_chain,my_llm

st.title("📄 PDF Q&A Bot using RAG")

pdf_file = st.file_uploader("Upload your PDF", type="pdf")

if pdf_file:
    with open("Data//pdf_file.name", "wb") as f:
        f.write(pdf_file.getbuffer())

    st.success("PDF uploaded. Creating knowledge base...")
    
    docs = load_and_split_pdf("Data//pdf_file.name")
    vector_store = create_vector_store(docs)
    llm=my_llm()
    qa_chain = get_qa_chain(vector_store,llm)

    query = st.text_input("Ask a question about your PDF:")
    if query:
        response = qa_chain.run(query)
        st.write("💬 Answer:", response)
