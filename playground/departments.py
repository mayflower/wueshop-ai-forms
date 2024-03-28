"""
Aufbau-Ideen-Klau:
https://python.langchain.com/docs/langgraph#multi-agent-examples

Hierachical: https://github.com/langchain-ai/langgraph/blob/main/examples/multi_agent/hierarchical_agent_teams.ipynb

Team-Conversation:>
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

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def create_agent(
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str,
) -> str:
    """Create a function-calling agent and add it to the graph."""
    system_prompt += "\nWork autonomously according to your specialty, using the tools available to you."
    " Do not ask for clarification."
    " Your other team members (and other teams) will collaborate with you with their own specialties."
    " You are chosen for a reason! You are one of the following team members: {team_members}."
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


def create_team_supervisor(llm: ChatOpenAI, system_prompt, members):
    """An LLM-based router."""
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    nextfunc_res = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )
    return nextfunc_res


def formatter(state):
    if state is None or len(state["messages"]) == 0:
        return "No Messages"
    return state["messages"][-1].content


class Berater:
    class BeraterState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        next: str
        current_answer: str

    def create_berater_graph():
        model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)

        def planer_agent(state: __class__.BeraterState):
            planer_tool = Planer.planer_invoker()
            messages = state["messages"]
            response = planer_tool.invoke(messages)
            return {
                "messages": messages
                + [SystemMessage(name="plan", content=response.content)],
                "current_answer": response,
            }

        def max_agent(state: __class__.BeraterState):
            max_tool = Max.max_invoker()
            messages = state["messages"]
            response = max_tool.invoke(messages)
            return {
                "messages": messages
                + [SystemMessage(name="redaktion", content=response.content)],
                "current_answer": response,
            }

        berater_agent = create_agent(
            model,
            [],
            "Du bist ein freundlicher Kundenberater. Frage den Planer nach Anweisungen, was zu tun ist.",
        )

        supervisor_router = create_team_supervisor(
            model,
            "Sie sind ein Supervisor, der mit der Leitung eines Gesprächs zwischen den"
            " folgenden Teams beauftragt ist: {team_members}. Angesichts der folgenden Benutzeranforderung,"
            " antworten Sie mit dem Arbeiter, der als nächstes handeln soll. Jeder Arbeiter wird eine"
            " Aufgabe ausführen und mit seinen Ergebnissen und Status antworten. Wenn fertig,"
            " antworten Sie mit FINISH."
            " Jeder Arbeiter wird nur einmal gefragt und antwortet mindestens einmal."
            " Du kannst einen Arbeiter nur ein mal fragen, wenn du mit seiner bisherigen Arbeit unzufrieden warst."
            " Bevor eine Antwort gegeben wird, muss die Redaktion mindestens ein mal angefragt worden sein. Auf jeden Fall zuletzt.",
            ["Planung", "Redaktion"],
        )

        def supervisor_node(state):
            return {
                **(supervisor_router.invoke({"messages": state["messages"]})),
                "messages": state["messages"],
            }

        workflow = StateGraph("beraterState")
        workflow.add_node("supervisor", supervisor_node)
        workflow.set_entry_point("supervisor")

        workflow.add_node("planer", planer_agent)
        workflow.add_edge("planer", "supervisor")

        workflow.add_node("redaktion", max_agent)
        workflow.add_edge("redaktion", "supervisor")

        # workflow.add_node("berater", berater_agent)
        # workflow.add_edge("berater", "supervisor")

        workflow.add_conditional_edges(
            "supervisor",
            lambda state: state["next"],
            {
                "Planung": "planer",
                "Redaktion": "redaktion",
                # "Kundenkontakt": "berater",
                "FINISH": END,
            },
        )

        graph = workflow.compile(debug=True)
        return graph

    def enter_chain(message: str):
        # TODO: Use supervisor thingies
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

    # TODO
    # def berater_invoker():
    #     return __class__.enter_chain | __class__.create_berater_graph()


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
