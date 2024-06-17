import os
from dotenv import load_dotenv
from langchain.storage import InMemoryStore
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain
from src.CustomParentDocumentRetriever import CustomParentDocumentRetriever

collection_name_parent_child = "wegleitung_parent_child"
collection_name_standard = "wegleitung_standard"
chroma_db_path_parent_child = "../chroma_db_parent_child"
chroma_db_path_standard = "../chroma_db_standard"

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

vectorstore_standard = None


def load_documents_parent_child():
    # delete the existing vectorstore
    vectorstore = Chroma(
        collection_name=collection_name_parent_child,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=chroma_db_path_parent_child,
    )
    try:
        vectorstore.delete_collection()
    except Exception as e:
        print(e)

    # Create it again to create a new collection
    vectorstore = Chroma(
        collection_name=collection_name_parent_child,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=chroma_db_path_parent_child,
    )

    loader = PyMuPDFLoader("../raw_data/Wegleitung_NP_2023.pdf")
    documents = loader.load()

    # This text splitter is used to create the parent documents of n chars
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    # This text splitter is used to create the child documents of n chars
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    # The storage layer for the parent documents
    docstore = InMemoryStore()

    retriever = CustomParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        search_kwargs={"k": 30},
        parent_splitter=parent_splitter,
    )

    #retriever = ParentDocumentRetriever(
    #    vectorstore=vectorstore,
    #    docstore=docstore,
    #    child_splitter=child_splitter,
    #    search_kwargs={"k": 30},
    #    parent_splitter=parent_splitter,
    #)

    # Performs the following steps:
    # - Splitting the documents into parent documents (4000 chars)
    # - Splitting the parent documents into child documents (400 chars)
    # - Embedding the child documents in the vectorstore
    # - Storing the parent documents in the docstore
    retriever.add_documents(documents, ids=None)

    print("Parent-Child Retriever - number of pdf pages loaded: " + str(len(documents)))
    print("Parent-Child Retriever - number of child docs: " + str(vectorstore._collection.count()))
    print("Parent-Child Retriever - number of parent docs: " + str(len(list(docstore.yield_keys()))))

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


@chain
def get_retriever(query: str) -> List[Document]:
    global vectorstore_standard

    docs, scores = zip(*vectorstore_standard.similarity_search_with_score(query, k=30))
    for doc, score in zip(docs, scores):
        doc.metadata["score"] = score

    return docs


def load_documents_standard():
    # delete the existing vectorstore
    global vectorstore_standard
    vectorstore_standard = Chroma(
        collection_name=collection_name_standard,
        embedding_function=OpenAIEmbeddings(),
        persist_directory=chroma_db_path_standard,
    )
    try:
        vectorstore_standard.delete_collection()
    except Exception as e:
        print(e)

    loader = PyMuPDFLoader("../raw_data/Wegleitung_NP_2023.pdf")
    documents = loader.load()

    # This text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(documents)

    vectorstore_standard = Chroma.from_documents(
        documents=splitted_docs,
        embedding=OpenAIEmbeddings(),
        collection_name=collection_name_standard,
        persist_directory=chroma_db_path_standard,
    )

    retriever = get_retriever

    print("Standard Retriever: number of splitted documents loaded: " + str(len(splitted_docs)))

    return retriever
