import csv
import os

from dotenv import load_dotenv
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.helper.functions import one_doc_per_pdf_page

file_path = '../raw_data/content-test.csv'
collection_name = "taxes_sg_child_documents"
chroma_db_path = "../chroma_db"

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def create_vectorstore():
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=chroma_db_path,
    )
    return vectorstore


def load_documents():
    # delete the existing vectorstore
    vectorstore = create_vectorstore()
    try:
        vectorstore.delete_collection()
    except Exception as e:
        print(e)

    # Create it again to create a new collection
    vectorstore = create_vectorstore()

    # Load the documents from the csv file
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        # Skip the header row
        next(reader)

        documents = []
        number_of_pdfs = 0
        for row in reader:
            # documents.append(one_doc_per_pdf(row))
            documents.extend(one_doc_per_pdf_page(row))
            number_of_pdfs += 1

    # This text splitter is used to create the parent documents of n chars
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    # This text splitter is used to create the child documents of n chars
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    # The storage layer for the parent documents
    docstore = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        # parent_splitter=parent_splitter,
    )

    # Performs the following steps:
    # - Splitting the documents into parent documents (4000 chars)
    # - Splitting the parent documents into child documents (400 chars)
    # - Embedding the child documents in the vectorstore
    # - Storing the parent documents in the docstore
    retriever.add_documents(documents, ids=None)

    print("Number of PDFs processed: " + str(number_of_pdfs))
    print("Number of documents loaded (one doc per PDF page): " + str(len(documents)))
    print("Number of parent docs: " + str(len(list(docstore.yield_keys()))))

    # Directly get the relevant parent documents (via implicit similarity search of sub-documents and then matching parent documents)
    # relevant_docs = retriever.invoke("Ich habe Eigenleistungen an meinem Grundstück erbracht, was muss ich beachten?")
    # print(relevant_docs)
    # print("Number of matching parent docs: " + str(len(relevant_docs)))
    # for doc in relevant_docs:
    #    print(doc.metadata)
    #    print("Number of chars: " + str(len(doc.page_content)))
    #    print(doc.page_content)

    # Just for curiosity, perform a similarity search on the vectorstore
    # sub_docs = vectorstore.similarity_search("Buchhaltung Landwirtschaft", k=6)
    # print("Number of similarity matching child documents: " + str(len(sub_docs)))
    # for d in sub_docs:
    #    print(d.metadata)

    return retriever
