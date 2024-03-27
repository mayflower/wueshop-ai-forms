from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain.schema import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessageGraph

load_dotenv()


def initialize_app():
    if "app" not in st.session_state:
        llm = ChatOpenAI()
        memory = SqliteSaver.from_conn_string(":memory:")
        workflow = MessageGraph()
        workflow.add_node("chatbot", llm)
        workflow.set_entry_point("chatbot")
        workflow.add_edge("chatbot", END)

        graph = workflow.compile(checkpointer=memory)

        def formatter(messages=List[BaseMessage]):
            return messages[-1].content

        app = graph | formatter

        st.session_state.app = app
