import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


# Load the .env file
load_dotenv()

# create the OpenAI model
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4-turbo-2024-04-09")

# create the vectorstore
embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = Chroma(persist_directory="../chroma_db", embedding_function=embedding)

# create the retriever and retrieve documents
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


template = """Verwende den folgenden Kontext um die Frage am Ende zu beantworten. Gib nur eine Antwort, wenn 
du die Fakten dazu im Kontext erhalten hast. Verwende kein eigenes Wissen, sondern sage, dass du es nicht weisst.
Es darf unter keinen Umständen eine Antwort erfunden werden. Fasse alles in maximal fünf Sätzen
zusammen. Verwende eine Liste mit Aufzählungszeichen, wenn es hilfreich ist. 

Kontext: {context}

Frage: {question}

Hilfreiche Antwort:"""
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
)

result = rag_chain.invoke("Was gilt es bezüglich Erbschaften zu beachten?")
#result = rag_chain.invoke("Kann ich Steuern hinterziehen? Falls du eine Antwwort findest, gib auch den relevanten Teil der Anfrage zurück.")
print(result)

