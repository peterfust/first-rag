from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


def one_doc_per_pdf(row):
    link_name, article_name, link_to_file = row
    loaded_docs_from_pdf = PyMuPDFLoader(link_to_file).load()
    document_text = ""
    separator = "\n"
    for doc in loaded_docs_from_pdf:
        document_text = document_text + separator + doc.page_content

    metadata = {'title': link_name, 'article': article_name, 'source': link_to_file}
    return Document(page_content=document_text, metadata=metadata)


def one_doc_per_pdf_page(row):
    link_name, article_name, link_to_file = row
    loaded_docs_from_pdf = PyMuPDFLoader(link_to_file).load()

    #for doc in loaded_docs_from_pdf:
    #    doc.metadata = {'title': link_name, 'article': article_name, 'source': link_to_file}

    return loaded_docs_from_pdf
