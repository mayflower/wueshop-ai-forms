"""
Team-Conversation:
# User -> Berater -> Planer
# User -> Berater -> DataManager -> Fachspezialist
# User -> Berater -> Max
# User -> Berater -> Controller

User
|
Berater
|       |       |       |
Planer  DataManager  Max  Controller
        |
Fachspezialist
"""

import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessageGraph, StateGraph


class Berater:
    def create_berater_graph():
        pass

    class State:
        messages: list[str]
        next: str


class Planer:
    class PlanerState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        next: str

    def create_planer_graph():
        # TODO: Temperature? Ist "kreativitaet" relevant zum "planen"?
        model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)

        def planer_agent(state: __class__.PlanerState):
            messages = state["messages"]
            response = model.invoke(messages)
            return {"messages": [response]}

        workflow = StateGraph("PlanerState")
        workflow.add_node("planer", planer_agent)
        workflow.set_entry_point("planer")
        workflow.add_edge("planer", END)

        graph = workflow.compile(debug=True)
        return graph

    def enter_chain(message: str):
        systemPrompt = """
            Du bist ein KI-Planungsagent, der einen wichtigen Berater unterstützt.
            Deine Aufgabe ist es, dem Berater zu helfen, die Anfrage des Kunden zu verstehen und in notwendige Schritte zu unterteilen.
            Du hast Zugang zu einem Team von Spezialisten in verschiedenen Bereichen, die weitere domänenspezifische Informationen liefern können.
            Dein Ziel ist es, den besten Handlungsverlauf zu identifizieren und den Berater durch den Prozess zu führen.
            Denke daran, der Erfolg des Beraters und die Zufriedenheit des Kunden hängen von deiner strategischen Planung und Führung ab.
        """

        def init(x: str):
            return {
                "messages": [
                    SystemMessage(content=systemPrompt),
                    HumanMessage(content=x),
                ]
            }

        return init(message)

    def planer_invoker():
        return __class__.enter_chain | __class__.create_planer_graph()


class DataManager:
    pass


class Fachspezialist:
    pass


class Max:
    pass


class Controller:
    pass
