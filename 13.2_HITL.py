# backend.py

from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
import requests

load_dotenv()


llm = ChatOpenAI()


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=TZ5VK41IOOOPSCE8"
    )
    r = requests.get(url)
    return r.json()

@tool     # we are using HITL in a TOOL
def purchase_stock(symbol: str, quantity: int) -> dict: #tools r bound to llm, it automatilcly pass these args
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    # This pauses the graph and returns control to the caller
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")
    # using isinctance check whether user type sting or num, then convert the str to lower
    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }


tools = [get_stock_price, purchase_stock]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)


memory = MemorySaver()


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=memory)

# Simple usage example (CLI with HITL)
if __name__ == "__main__":
    thread_id = "demo-thread"

    while True:
        user_input = input("You: ")
        if user_input.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break

        state = {"messages": [HumanMessage(content=user_input)]}

        result = chatbot.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )

        # This checks whether the graph execution paused because of an interrupt() call.
        interrupts = result.get("__interrupt__", []) # very imp step

        if interrupts:
            prompt_to_human = interrupts[0].value # this means ==> interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")
            print(f"HITL: {prompt_to_human}")
            decision = input("Your decision: ").strip().lower()

            result = chatbot.invoke(
                Command(resume=decision),
                config={"configurable": {"thread_id": thread_id}},
            )

        messages = result["messages"]
        last_msg = messages[-1]
        print(f"Bot: {last_msg.content}\n")