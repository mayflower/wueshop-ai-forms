import streamlit as st
from langchain_core.messages import HumanMessage

from form_helper import initialize_app
from pdf_loader import loader

initialize_app()
config = {"configurable": {"thread_id": "1"}}
st.title("Form Helper")


with st.form("chatbot_form"):
    text = st.text_area(
        "Enter text:",
        "Please enter your question to the LLM here.",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.session_state.current_answer = st.session_state.app.invoke(
            {"messages": [HumanMessage(content=text)]}, config=config
        )
    if "current_answer" in st.session_state:
        st.info(st.session_state.current_answer)

load_button = st.button("Load Document")
if load_button:
    load_result = loader.load()
    st.text_area("Load Result:", value=str(load_result))
