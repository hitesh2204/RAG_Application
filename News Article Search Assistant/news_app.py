import streamlit as st
from ingest import load_and_split, create_vector_store, get_qa_chain,my_llm

st.title("📄 News chat-bot")

text_file = st.file_uploader("Upload your text file", type=["txt", "pdf"])

if text_file:
    with open(f"Data/{text_file.name}", "wb") as f:
        f.write(text_file.getbuffer())
    st.success("PDF uploaded. Creating knowledge base...")

    docs =  load_and_split(f"Data/{text_file.name}")

    vector_store = create_vector_store(docs)
    llm=my_llm()
    qa_chain = get_qa_chain(vector_store,llm)

    query = st.text_input("Ask a question about your PDF:")
    if query:
        response = qa_chain.run(query)
        st.write("💬 Answer:", response)
