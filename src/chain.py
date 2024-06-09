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

# create a reranker with cohere
cohere_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(cohere_key)


def format_docs(docs_and_question):
    print("Length of docs to format: ", len(docs_and_question['context']))
    docs = "\n\n".join(doc.page_content for doc in docs_and_question['context'])
    return {"context": docs, "question": docs_and_question['question']}


def rerank_docs_cohere(docs_and_question):
    print("Number of docs BEFORE reranking: ", len(docs_and_question['context']))
    docs_for_rerank = [doc.page_content for doc in docs_and_question['context']]
    response = co.rerank(
        model="rerank-multilingual-v3.0",
        query=docs_and_question['question'],
        documents=docs_for_rerank,
        top_n=5,
    )
    print("Number of docs AFTER reranking: ", len(response.results))
    reranked_docs = [docs_and_question['context'][res.index] for res in response.results]
    docs_and_question['context'] = reranked_docs
    return docs_and_question


def chain(retriever):
    template = """Du bist ein hilfreicher Assistent und beantwortest Fragen zum Steuerbuch in einem höflichen Ton. 
    Fasse alles in präzise und in klaren Worten zusammen. Verwende eine Liste mit Aufzählungszeichen, aber nur,
    wenn es hilfreich ist. Verwende den folgenden Kontext um die Frage am Ende zu beantworten. Gib nur eine Antwort, 
    wenn du die Fakten dazu im Kontext erhalten hast. Verwende kein eigenes Wissen, sondern sage, dass du es nicht 
    weisst. Es darf unter keinen Umständen eine Antwort erfunden werden. 
    
    Kontext: {context}
    
    Frage: {question}
    
    Hilfreiche Antwort:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            # | RunnableLambda(rerank_docs_cohere)
            | RunnableLambda(format_docs)
            | custom_rag_prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain

