import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, LLMChain
from langchain import hub
from langchain.callbacks import StdOutCallbackHandler
from langchain_community.vectorstores import FAISS
from langchain.agents import Tool, ZeroShotAgent, AgentExecutor, create_react_agent
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

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)

retrieval_chain_2010 = create_retrieval_chain(retriever_2010, combine_docs_chain)
retrieval_chain_basics = create_retrieval_chain(retriever_basics, combine_docs_chain)

def query_2010(input):
    return retrieval_chain_2010.invoke({'input': input})

def query_basics(input):
    return retrieval_chain_basics.invoke({'input': input})

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

prompt = hub.pull("hwchase17/react")
excel_agent = create_react_agent(llm, tools, prompt)
excel_agent_chain = AgentExecutor.from_agent_and_tools(agent=excel_agent, tools=tools, verbose=True, return_intermediate_steps=True)

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
    result = excel_agent_chain.invoke({'input':message.content})

    question = f"Initial query... {result['input']}\n\n"
    response = f"Final response... {result['output']}\n\n"
    hold = ""
    for i in range(len(result['intermediate_steps'])):
        reflection = result['intermediate_steps'][i][0].log.split('\n')[0]
        tool = result['intermediate_steps'][i][0].tool
        tool_in = result['intermediate_steps'][i][0].tool_input
        resp = result['intermediate_steps'][i][1]['answer']
        hold += f'Step {i+1}:\n'
        hold += f'The agent considers... "{reflection}"\n'
        hold += f'The angent chooses the tool...  "{tool}."\n'
        hold += f'The angent queries the tool... "{tool_in}."\n'
        hold += f'The angent recieves the response...\n"{resp}"\n\n'

    print_response = question + response + hold
    await cl.Message(print_response).send()