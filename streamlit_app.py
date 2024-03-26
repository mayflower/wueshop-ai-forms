import streamlit as st

from form_helper import llm

st.title("Form Helper")


with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "Enter your question to the LLM here.",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        answer = llm.invoke(text).content
        st.info(answer)
