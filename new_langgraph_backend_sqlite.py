from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal 
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
import operator
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import add_messages # use add_messages (reducer func) for list of BaseMessage
import sqlite3
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool




os.environ['LANGCHAIN_PROJECT'] = 'Streamlit_Sqlite_Chatbot' # making it in a new project
load_dotenv()
llm = ChatOpenAI(model="gpt-4.1-nano")


# **************** Tools **********************************
search_tool = DuckDuckGoSearchRun(region="us-en")

@tool
def calculator(first_num: float, second_num: float, operation: str):
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}

tools = [search_tool, calculator]
llm_with_tools = llm.bind_tools(tools)


# **************** SQLITE Db **********************************
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
checkpointer = SqliteSaver(conn = conn)



# ********************* STATE and Graph *****************************
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]    # basemessage is the parent class of humanmessage, systemmessage and aimessage

def chat_node(state: ChatState):
    """
    LLM node that may answer or request a tool call.
    """
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)  

graph = StateGraph(ChatState)
#graph.add_node('chat_node', chat_node)
#graph.add_edge(START, 'chat_node')
#graph.add_edge('chat_node', END)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)# mostly put all the tools to be used in one tool node only
graph.add_edge("tools", "chat_node")


chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None): # i want all, and not for specific
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)







# test
#CONFIG = {'configurable':{"thread_id":"thread_1"}}
#response = chatbot.invoke(
#    {'messages':[HumanMessage(content='what is my name')]},
#    config = CONFIG
#    )
#print(response)