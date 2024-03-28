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


def formatter(state):
    if state is None or len(state["messages"]) == 0:
        return "No Messages"
    return state["messages"][-1].content


class Berater:
    class BeraterState:
        messages: Annotated[List[BaseMessage], operator.add]
        next: str
        current_answer: str

    def create_berater_graph():
        planer_tool = Planer.planer_invoker()
        max_tool = Max.planer_invoker()

        def berater_agent(state: __class__.BeraterState):
            messages = state["messages"]
            response = planer_tool.invoke(messages)
            return {"messages": [response], "current_answer": response}

        workflow = StateGraph("beraterState")
        workflow.add_node("berater", berater_agent)
        workflow.set_entry_point("berater")
        workflow.add_edge("berater", END)

        graph = workflow.compile(debug=True)
        return graph

    def enter_chain(message: str):
        systemPrompt = """
            Du bist ein freundlicher Kundenberater. Frage den Planer nach Anweisungen, was zu tun ist.
        """

        def init(x: str):
            return {
                "messages": [
                    SystemMessage(content=systemPrompt),
                    HumanMessage(content=x),
                ]
            }

        return init(message)

    def berater_invoker():
        return __class__.enter_chain | __class__.create_berater_graph()


class Planer:
    class PlanerState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        next: str

    systemPrompt = """
            Du bist ein KI-Planungsagent, der einen wichtigen Berater unterstützt.
            Deine Aufgabe ist es, dem Berater zu helfen, die Anfrage des Kunden zu verstehen und in notwendige Schritte zu unterteilen.
            Du hast Zugang zu einem Team von Spezialisten in verschiedenen Bereichen, die weitere domänenspezifische Informationen liefern können.
            Dein Ziel ist es, den besten Handlungsverlauf zu identifizieren und den Berater durch den Prozess zu führen.
            Denke daran, der Erfolg des Beraters und die Zufriedenheit des Kunden hängen von deiner strategischen Planung und Führung ab.
            Antworte in einer Liste einfacher Schritte.
            (Beispiel: 1. Geheimzahl abfragen 2. Fachspezialisten fragen 3. …)
        """

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

    def enter_chain_direct(message: str):
        def init(x: str):
            return {
                "messages": [
                    SystemMessage(content=__class__.systemPrompt),
                    HumanMessage(content=x),
                ]
            }

        return init(message)

    def enter_chain(messages: List[BaseMessage]):
        def init(x: List[BaseMessage]):
            return {"messages": [SystemMessage(content=__class__.systemPrompt)] + x}

        return init(messages)

    def extract_answer(planer_state: dict):
        return planer_state["messages"][-1]

    def planer_invoker():
        return (
            __class__.enter_chain
            | __class__.create_planer_graph()
            | __class__.extract_answer
        )


class DataManager:
    pass


class Fachspezialist:
    pass


class Max:
    class MaxState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        next: str

    systemPrompt = """
            Du bist ein Experte für Kommunikation und musst Texte oder Listen in eine einfache und übersichtliche Form bringen.
            Du hilfst einem Berater eine Kundenanfrage zu bearbeiteten und wirst dazu Informationen von einem Planer erhalten.
            Gib dir Mühe die relevanten Informationen klar darzulegen. Wenn du das Gefühl hast, dass dir Infos fehlen, frage nach.
            Denke daran, der Erfolg des Beraters und die Zufriedenheit des Kunden hängen von deinen Formulierungen der Antwort ab.
        """

    def create_max_graph():
        model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)

        def max_agent(state: __class__.MaxState):
            messages = state["messages"]
            response = model.invoke(messages)
            return {"messages": [response]}

        workflow = StateGraph("MaxState")
        workflow.add_node("max", max_agent)
        workflow.set_entry_point("max")
        workflow.add_edge("max", END)

        graph = workflow.compile(debug=True)
        return graph

    def enter_chain_direct(message: str):
        def init(x: str):
            return {
                "messages": [
                    SystemMessage(content=__class__.systemPrompt),
                    HumanMessage(content=x),
                ]
            }

        return init(message)

    def enter_chain(messages: List[BaseMessage]):
        def init(x: List[BaseMessage]):
            return {"messages": [SystemMessage(content=__class__.systemPrompt)] + x}

        return init(messages)

    def extract_answer(max_state: dict):
        return max_state["messages"][-1]

    def max_invoker():
        return (
            __class__.enter_chain
            | __class__.create_max_graph()
            | __class__.extract_answer
        )


class Controller:
    pass
