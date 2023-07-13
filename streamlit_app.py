import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceHubEmbeddings

import os
token = os.environ.get('HUGGINGFACEHUB_API_TOKEN')

# Add a title and description for your app
st.set_page_config(
    page_title="SocialWorkSG_GPT",
    page_icon=":mag:",
    layout="centered",
    initial_sidebar_state="auto"
)

def load_document(file_path):
    loader = PyPDFLoader(file_path)
    return loader

def create_index(loader):
    model_name = "google/flan-t5-xl"
    embeddings = HuggingFaceHubEmbeddings(
        huggingfacehub_api_token=token
        #model_name=model_name
        )

    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=Chroma,
        embedding=embeddings,
        text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    )

    index = index_creator.from_loaders([loader])
    return index

def perform_query(index, query):
    results = index.query(query)
    return results

def main():
    st.title("LangChain Document Search")

    # Load the PDF document
    document = load_document('SASW-Code-of-Professional-Ethics-3rd-Revision-online.pdf')
    index = create_index(document)

    # Query input
    query = st.text_input("Enter your query")

    # Search button
    if st.button("Search"):
        results = perform_query(index, query)
        st.write("Search Results:")
        for result in results:
            st.write(result)

if __name__ == "__main__":
    main()
