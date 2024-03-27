import sqlite3
from typing import List
import json
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessageGraph
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain.globals import set_debug
from langgraph.prebuilt import ToolInvocation, ToolExecutor
from langchain_core.messages import ToolMessage
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.memory import ConversationBufferWindowMemory
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_community.document_loaders.text import TextLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_core.documents import Document


db = None
@tool
def geheimzahl_tool():
    """Das ist dein Tool - nutze es nur für die Geheimzahl
    Dieses Tool nimmt keine Parameter, also übergib bitte keine Argumente."""
    print("invoking geheimzahl tool")
    return "Die Geheimzahl ist 123987"

@tool
def document_tool(message: Annotated[str, "Eine Zusammenfassung der Vorhaben des Nutzers"]) -> Annotated[List[Document], "Eine Liste aller gefundenen Dokumente"]:
    """Wenn nach einem Fest oder Veranstaltung gefragt wird, nutze dieses Tool. Teile dem Nutzer immer mit welche Dateien wichtig sind."""
    docs_and_scores = db.similarity_search_with_score(message)
    print(len(docs_and_scores))
    return_docs = []
    for doc in docs_and_scores:
        print(doc)
        return_docs.append(doc)

    return return_docs


def rag_initialize():
    embeddings = OpenAIEmbeddings()


    loader = TextLoader("./playground/state_of_the_union.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    global db
    db = FAISS.from_documents(docs, embeddings)
    #print(db.index.ntotal)

    loader = PyPDFLoader(file_path="./playground/schankerlaubnis.pdf")
    documents = loader.load()
    #text = pages[0].page_content
    #db = FAISS.from_documents(docs, embeddings)
    db.add_documents(documents=documents)
    

ddgs = DuckDuckGoSearchRun()
tools = [geheimzahl_tool, document_tool]


load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def initialize_app():
    if "app" not in st.session_state:
        set_debug(True)
        rag_initialize()
        llm = ChatOpenAI()
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        model_with_tools = llm.bind_tools(tools)
        memory = SqliteSaver(conn=conn)
        def agent(state):
            print("invoking agent", state)
            messages = state["messages"]
            response = model_with_tools.invoke(messages)
            print(messages, response)
            return {"messages": [response]}

        def tool(state):
            print("invoking tool", state)
            messages = state["messages"][-1]
            tool_messages = []
            for tool_call in messages.additional_kwargs["tool_calls"]:
                action = ToolInvocation(
                    tool=tool_call["function"]["name"],
                    tool_input=json.loads(tool_call["function"]["arguments"] or {}),
                )
                response = tool_executor.invoke(action)
                tool_messages.append(
                    ToolMessage(
                        content=str(response),
                        tool_call_id=tool_call["id"],
                        name=tool_call["function"]["name"],
                    )
                )
            return {"messages": tool_messages}

        workflow = StateGraph(AgentState)
        workflow.add_node("chatbot", agent)
        workflow.set_entry_point("chatbot")
        workflow.add_node("search", tool)
        workflow.add_conditional_edges(
            "chatbot", should_continue, {"continue": "search", "end": END}
        )
        workflow.add_edge("search", "chatbot")

        graph = workflow.compile(checkpointer=memory)

        def formatter(state: List[BaseMessage]):
            if state is None or len(state["messages"]) == 0:
                return "No Messages"
            return state["messages"][-1].content

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
