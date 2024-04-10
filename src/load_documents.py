import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

pdf_folder_path = "../raw_data"
documents = []
for file in os.listdir(pdf_folder_path):
    if file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder_path, file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
chunked_docs = text_splitter.split_documents(documents)
print("Number of chunks loaded into vectorstore: " + str(len(chunked_docs)))

vectorstore = Chroma.from_documents(documents=chunked_docs,
                                    embedding=OpenAIEmbeddings(),
                                    persist_directory="../chroma_db")