import os

import cohere
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
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
    print("Cohere - Number of docs BEFORE reranking: ", len(docs_and_question['context']))
    docs_for_rerank = [doc.page_content for doc in docs_and_question['context']]
    response = co.rerank(
        model="rerank-multilingual-v3.0",
        query=docs_and_question['question'],
        documents=docs_for_rerank,
        top_n=5,
    )
    print("Cohere - Number of docs AFTER reranking: ", len(response.results))
    for res in response.results:
        metadata = docs_and_question['context'][res.index].metadata
        print(str(res) + ' - ' + metadata['source'] + '__' + str(metadata['page']))
    reranked_docs = [docs_and_question['context'][res.index] for res in response.results]
    docs_and_question['context'] = reranked_docs
    return docs_and_question


def create_reranking_prompt(docs_and_question):
    documents = "\n".join(f"-===START==={doc.metadata['source']}__{doc.metadata['page']}  {doc.page_content} ===ENDE===" for i, doc in enumerate(docs_and_question['context']))

    rerank_template = f"""Gestellte Frage: {docs_and_question['question']}

    Dokumente:
    {documents}
    
    Anweisung: Bitte bewerte die Relevanz jedes der oben aufgeführten Dokumente im Hinblick auf die gestellte Frage. 
    Berücksichtige dabei, wie gut jedes Dokument die Frage beantwortet, ob es relevante Informationen enthält und ob 
    diese Informationen korrekt und vertrauenswürdig erscheinen. Die einzelnen Dokumente sind mit ===START=== und 
    ===ENDE=== markiert. Der Name des Dokuments steht gleich nach dem Marker ===START===, danach folgt der Inhalt des
    Dokuments.
    Gib als Antwort ein JSON array zurück mit einer Liste von JSON Objekten. Jedes Objekt enthält die folgenden Felder:
    - name: Name des Dokuments
    - relevant: true, wenn das Dokument relevant ist, false, wenn nicht
    """

    rerank_llm_prompt = PromptTemplate.from_template(rerank_template)
    return rerank_llm_prompt


def postprocess_reranking(data):
    print("LLM Reranking - Number of documents BEFORE reranking: ", len(data['rerank']))
    print("LLM Reranking - Documents marked as relevant: ", [doc['name'] for doc in data['rerank'] if doc['relevant']])
    print("LLM Reranking - Documents marked as not relevant: ", [doc['name'] for doc in data['rerank'] if not doc['relevant']])
    relevant_docs_and_question = {'context': [], 'question': data['forward']['question']}
    for item in data['rerank']:
        if item['relevant']:
            for doc in data['forward']['context']:
                if item['name'] == doc.metadata['source'] + '__' + str(doc.metadata['page']):
                    temp = Document(page_content=doc.page_content, metadata=doc.metadata)
                    relevant_docs_and_question['context'].append(temp)
                    break
    print("LLM Reranking - Number of documents AFTER reranking: ", len(relevant_docs_and_question['context']))
    return relevant_docs_and_question


def chain(retriever):
    llm_template = """Du bist ein hilfreicher Assistent und beantwortest Fragen zum Steuerbuch in einem höflichen Ton. 
    Fasse alles in präzise und in klaren Worten zusammen. Verwende eine Liste mit Aufzählungszeichen, aber nur,
    wenn es hilfreich ist. Verwende den folgenden Kontext um die Frage am Ende zu beantworten. Gib nur eine Antwort, 
    wenn du die Fakten dazu im Kontext erhalten hast. Verwende kein eigenes Wissen, sondern sage, dass du es nicht 
    weisst. Es darf unter keinen Umständen eine Antwort erfunden werden. 
    
    Kontext: {context}
    
    Frage: {question}
    
    Hilfreiche Antwort:"""
    llm_prompt = PromptTemplate.from_template(llm_template)

    llm_rerank_chain = (
        RunnableLambda(create_reranking_prompt)
        | llm
        | JsonOutputParser()
    )

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | RunnableParallel(rerank=llm_rerank_chain, forward=RunnablePassthrough()) | RunnableLambda(postprocess_reranking)
            #| RunnableLambda(rerank_docs_cohere)
            | RunnableLambda(format_docs)
            | llm_prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain
