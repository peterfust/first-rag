import os
import re
import uuid
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
    print("Number of docs prepared for LLM: ", len(docs_and_question['context']))
    for doc in docs_and_question['context']:

        # from standard retriever
        if 'score' in doc.metadata:
            print(doc.metadata['source'] + '__' + str(doc.metadata['page']) + ', score: ' + str(doc.metadata['score']))

        # from parent-child retriever
        if 'sub_docs' in doc.metadata:
            print(doc.metadata['source'] + '__' + str(doc.metadata['page']))
            for sub_doc in doc.metadata['sub_docs']:
                print('- score: ' + str(sub_doc.metadata['score']))

    docs = "\n\n".join(doc.page_content for doc in docs_and_question['context'])
    return {"context": docs, "question": docs_and_question['question']}


def rerank_docs_cohere(docs_and_question):
    print("Cohere - Number of docs BEFORE reranking: ", len(docs_and_question['context']))
    docs_for_rerank = [doc.page_content for doc in docs_and_question['context']]
    response = co.rerank(
        model="rerank-multilingual-v3.0",
        query=docs_and_question['question'],
        documents=docs_for_rerank,
        top_n=len(docs_and_question['context']),  # we want to rerank all documents and then filter per threshold
    )

    reranked_docs = []
    for res in response.results:
        metadata = docs_and_question['context'][res.index].metadata
        print(str(res) + ' - ' + metadata['source'] + '__' + str(metadata['page']))
        if res.relevance_score >= 0.5:
            reranked_docs.append(docs_and_question['context'][res.index])
    docs_and_question['context'] = reranked_docs
    print("Cohere - Number of docs AFTER reranking: ", len(reranked_docs))
    return docs_and_question


def create_llm_prompt(docs_and_question):
    llm_template = f"""Du bist ein ein äusserst hilfreicher und genauer Assistent und beantwortest Steuerfragen in 
    einem höflichen Ton. Fasse alles in präzise und in klaren Worten zusammen. Verwende den nachfolgenden Kontext um 
    die Frage am Ende zu beantworten. Gib nur eine Antwort, wenn du die Fakten dazu im Kontext erhalten hast. 
    Verwende kein eigenes Wissen, sondern sage, dass du es nicht weisst. Verwende HTML, um die Antwort zu 
    formatieren. 
    
    Kontext: {docs_and_question['context']}
    
    Frage: {docs_and_question['question']}
    """
    llm_prompt = PromptTemplate.from_template(llm_template)
    return llm_prompt


def create_reranking_prompt(docs_and_question):
    print("OpenAI Reranking - Number of documents BEFORE reranking: ", len(docs_and_question['context']))
    documents = "\n".join(f"-===START=== ==UUID_START=={doc.metadata['uuid']}==UUID_END==  {doc.page_content} ===ENDE===" for i, doc in enumerate(docs_and_question['context']))

    rerank_template = f"""Gestellte Frage: {docs_and_question['question']}

    Dokumente:
    {documents}
    
    Anweisung: Bitte bewerte die Relevanz jedes der oben aufgeführten Dokumente im Hinblick auf die gestellte Frage. 
    Berücksichtige dabei, wie gut jedes Dokument die Frage beantwortet, ob es relevante Informationen enthält und ob 
    diese Informationen korrekt und vertrauenswürdig erscheinen. Die einzelnen Dokumente sind mit ===START=== und 
    ===ENDE=== markiert. Die UUID des Dokuments steht gleich nach dem Marker ===START=== und wird mit ==UUID_START== 
    und ==UUID_END== markiert. Danach folgt der Inhalt des Dokuments.
    Gib als Antwort ein JSON array zurück mit einer Liste von JSON Objekten. Jedes Objekt enthält die folgenden Felder:
    - uuid: UUID des Dokuments
    - relevant: true, wenn das Dokument relevant ist, false, wenn nicht
    """

    rerank_llm_prompt = PromptTemplate.from_template(rerank_template)
    return rerank_llm_prompt


def postprocess_openai_reranking(data):
    relevant_docs = [doc['uuid'] for doc in data['rerank'] if doc['relevant']]
    not_relevant_docs = [doc['uuid'] for doc in data['rerank'] if not doc['relevant']]
    print("OpenAI Reranking - Documents marked as relevant: ", len(relevant_docs))
    print("OpenAI Reranking - Documents marked as relevant: ", relevant_docs)
    print("OpenAI Reranking - Documents marked as not relevant: ", len(not_relevant_docs))
    print("OpenAI Reranking - Documents marked as not relevant: ", not_relevant_docs)
    relevant_docs_and_question = {'context': [], 'question': data['forward']['question']}
    for item in relevant_docs:
        for doc in data['forward']['context']:
            if item == doc.metadata['uuid']:
                temp = Document(page_content=doc.page_content, metadata=doc.metadata)
                relevant_docs_and_question['context'].append(temp)
                break
    print("OpenAI Reranking - Number of documents AFTER reranking: ", len(relevant_docs_and_question['context']))
    return relevant_docs_and_question


def add_unique_doc_id(docs_and_question):
    docs_and_question_new = docs_and_question
    for i, doc in enumerate(docs_and_question['context']):
        doc.metadata['uuid'] = str(uuid.uuid4())
    return docs_and_question_new


def chain_with_cohere_reranking(retriever):
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | RunnableLambda(rerank_docs_cohere)
        | RunnableLambda(format_docs)
        | RunnableLambda(create_llm_prompt)
        | llm
        | StrOutputParser()
    )

    return chain


def chain_with_openai_reranking(retriever):
    openai_rerank_chain = (
            RunnableLambda(create_reranking_prompt)
            | llm
            | JsonOutputParser()
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | RunnableLambda(add_unique_doc_id)
        | RunnableParallel(rerank=openai_rerank_chain, forward=RunnablePassthrough()) | RunnableLambda(postprocess_openai_reranking)
        | RunnableLambda(format_docs)
        | RunnableLambda(create_llm_prompt)
        | llm
        | StrOutputParser()
    )

    return chain


def chain_standard(retriever):

    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | RunnableLambda(format_docs)
            | RunnableLambda(create_llm_prompt)
            | llm
            | StrOutputParser()
    )

    return rag_chain
