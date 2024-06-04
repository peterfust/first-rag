import os

import cohere
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

# Load the .env file
load_dotenv()

# create the OpenAI model
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o", temperature=0)


def format_docs(docs):
    print("Length of docs: ", len(docs))
    for doc in docs:
        print(doc.metadata)
        print(doc.page_content)
    return "\n\n".join(doc.page_content for doc in docs)


def print_input(input):
    print(input)
    return input


# create a reranker with cohere
cohere_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(cohere_key)


def rerank_docs(docs):
    if docs['context']:
        docs_for_rerank = [doc.page_content for doc in docs['context']]

        response = co.rerank(
            model="rerank-multilingual-v3.0",
            query=docs['question'],
            documents=docs_for_rerank,
            top_n=3,
        )

        reranked_docs = [docs['context'][res.index] for res in response.results]

        docs['context'] = reranked_docs
        return docs
    else:
        return docs


def chain(retriever):
    template = """Du bist ein hilfreicher Assistent und beantwortest Fragen zum Steuerbuch. Fasse alles in präzise und in klaren Worten
    zusammen. Verwende eine Liste mit Aufzählungszeichen, aber nur wenn es hilfreich ist. 
    Verwende den folgenden Kontext um die Frage am Ende zu beantworten. Gib nur eine Antwort, wenn 
    du die Fakten dazu im Kontext erhalten hast. Verwende kein eigenes Wissen, sondern sage, dass du es nicht weisst.
    Es darf unter keinen Umständen eine Antwort erfunden werden. 
    
    Kontext: {context}
    
    Frage: {question}
    
    Hilfreiche Antwort:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | RunnableLambda(rerank_docs)
            | custom_rag_prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain

