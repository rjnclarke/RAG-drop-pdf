import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os 

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model='gpt-3.5-turbo')
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("dbs/excel_train.db",embeddings,allow_dangerous_deserialization=True)

@cl.on_chat_start
async def on_chat_start():
    content = 'This is a chat app for asking about the new changes to Excel 2010\n\n'
    user_prompt = 'Please type a question...'
    await cl.Message(content + user_prompt).send()

@cl.on_message
async def on_message(message: cl.Message):
    # getting db and retriever
    
    retriever = db.as_retriever()
    handler = StdOutCallbackHandler()

    qa_with_sources_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        callbacks=[handler],
        return_source_documents=True
    )

    response = qa_with_sources_chain({"query" : message.content})
    query = response['query']
    result = response['result']
    source_docs = response['source_documents'][0].page_content





    response = f"You Sent: {query}\n\nResponse: {result}\n\nSource: {source_docs}!"
    await cl.Message(response).send()