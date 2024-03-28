import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from form_helper import initialize_app
from pdf_loader import loader
from playground.departments import Berater

initialize_app()

# memory = SqliteSaver.from_conn_string(":memory:")
config = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}
st.title("Form Helper")


with st.form("chatbot_form"):
    text = st.text_area(
        "Enter text:",
        "Ich will ein Fest feiern und Alkohol trinken.",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.session_state.current_answer = st.session_state.app.invoke(
            {
                "messages": [
                    SystemMessage(
                        content="Du bist Mitarbeiter im Bürgerbüro der Stadt Würzburg und Zuständig für die Dokumentenverwaltung. Es ist von absolut kritischer Wichtigkeit, dass alle informationen die du herausgibst aus deinen zur Verfügung stehenden Tools stammen!"
                    ),
                    HumanMessage(content=text),
                ]
            },
            config=config,
        )

    if "current_answer" in st.session_state:
        st.info(st.session_state.current_answer)

with st.form("berater_form"):

    def formatter(state):
        if state is None or len(state["messages"]) == 0:
            return "No Messages"
        return state["messages"][-1].content

    submitted = st.form_submit_button("Submit Berater")
    berater_invoke = Berater.berater_invoker() | formatter

    if "berater" not in st.session_state:
        st.session_state["berater"] = {
            "app": berater_invoke,
            "current_answer": None,
        }

    text = st.text_area(
        "berater:",
        "Was muss ich tun, um in Würzburg ein Fest zu feiern? Antworte in einem Satz!",
    )
    if submitted:
        st.session_state.berater["current_answer"] = st.session_state.berater[
            "app"
        ].invoke(
            text,
            config=config,
        )
    if (
        "current_answer" in st.session_state.berater
        and st.session_state.berater.get("current_answer") is not None
    ):
        st.info(st.session_state.berater["current_answer"])

load_button = st.button("Load Document")
if load_button:
    load_result = loader.load()
    st.text_area("Load Result:", value=str(load_result))
