from typing import List
import json
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessageGraph
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain.globals import set_debug
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from langchain_core.messages import ToolMessage
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph


ddgs = DuckDuckGoSearchRun()
tools = [ddgs]

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def initialize_app():
    if "app" not in st.session_state:
        set_debug(True)
        llm = ChatOpenAI()
        model_with_tools = llm.bind_tools(tools)
        memory = SqliteSaver.from_conn_string(":memory:")

        def agent(state):
            print("invoking agent", state)
            messages = state["messages"]
            response = model_with_tools.invoke(messages)
            return {"messages": [response]}

        def tool(state):
            print("invoking tool", state)
            messages = state["messages"][-1]
            tool_call = messages.additional_kwargs["tool_calls"][0]
            action = ToolInvocation(
                tool=tool_call["function"]["name"],
                tool_input=json.loads(tool_call["function"]["arguments"] or {}),
            )
            response = tool_executor.invoke(action)
            tool_message = ToolMessage(
                content=str(response),
                tool_call_id=tool_call["id"],
                name=tool_call["function"]["name"],
            )
            return {"messages": [tool_message]}

        workflow = StateGraph(AgentState)
        workflow.add_node("chatbot", agent)
        workflow.set_entry_point("chatbot")
        workflow.add_node("search", tool)
        workflow.add_conditional_edges(
            "chatbot", should_continue, {"continue": "search", "end": END}
        )
        workflow.add_edge("search", "chatbot")

        graph = workflow.compile()

        def formatter(messages=List[BaseMessage]):
            if len(messages):
                return "No Messages"
            return messages[-1].content

        app = graph | formatter

        st.session_state.app = app

        tool_executor = ToolExecutor(tools)


def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if "tool_calls" not in last_message.additional_kwargs:
        return "end"
    else:
        return "continue"
