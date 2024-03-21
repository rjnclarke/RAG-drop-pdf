import chainlit as cl
from langchain.chains import RetrievalQA, LLMChain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
from langchain_community.vectorstores import FAISS
from langchain.agents import Tool, ZeroShotAgent, AgentExecutor
from dotenv import load_dotenv
import os 

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model='gpt-3.5-turbo')
embeddings = OpenAIEmbeddings()
handler = StdOutCallbackHandler()

db_excel_2010 = FAISS.load_local("dbs/excel_train.db",embeddings,allow_dangerous_deserialization=True)
db_excel_basics = FAISS.load_local("dbs/excel_basics.db",embeddings,allow_dangerous_deserialization=True)
retriever_2010 = db_excel_2010.as_retriever()
retriever_basics = db_excel_basics.as_retriever()

qa_with_sources_chain_2010 = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever_2010,
    callbacks=[handler],
    return_source_documents=True 
)

qa_with_sources_chain_basics = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever_basics,
    callbacks=[handler],
    return_source_documents=True
)

def query_2010(input):
    return qa_with_sources_chain_2010('query', input)

def query_basics(input):
    return qa_with_sources_chain_basics('query', input)

tools = [
    Tool(
        name = "Excel2010",
        func= query_2010,
        description="Useful for understanding the new features released in MS Excel 2010."
    ),
    Tool(
        name = "ExcelBasics",
        func=query_basics,
        description="Useful for explaining how to perform basic operations in MS Excel."
    ),
]

prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"

Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "agent_scratchpad"]
)


llm_chain = LLMChain(llm=llm, prompt=prompt)

excel_agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
excel_agent_chain = AgentExecutor.from_agent_and_tools(agent=excel_agent, tools=tools, verbose=True)

@cl.on_chat_start
async def on_chat_start():
    content = "This is a chat app relating to excel. The LLM can draw on two resources:\n\n\
    1: A document explaining how to use the basic functionality of Excel.\n\
    2: A document explaining the new features of Excel 2010\n\n\
    When you question the LLM, it will decide if it needs to rely on any of these resources or not.\n\
    The response will show the deliberations of the LLM and cite any sources in the response."

    user_prompt = 'Please ask a question...'

    await cl.Message(content + user_prompt).send()

@cl.on_message
async def on_message(message: cl.Message):
    # getting db and retriever
    response = excel_agent_chain.run({"input" : message.content})
    





    reply = f"You Sent: {message.content}\n\nResponse: {response}!"
    await cl.Message(reply).send()