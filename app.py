import streamlit as st
import pandas as pd
import numpy as np
import ai

import gc
gc.collect()

default_messages = [{"role": "assistant", "content": "How may I assist you today?"}]

st.title("Jujutsu Kaisen QA")

docs = None

def get_lang(lang):
    if lang == "Portuguese":
        return "pt"
    else:
        return "en"

with st.sidebar:
    language = st.radio("Select Language", ["Portuguese", "English"])
    lang = get_lang(language)

    repo_id = st.selectbox(
        "Select Model",
        [
            "google/flan-t5-small",
            "google/flan-t5-large",
            "HuggingFaceH4/zephyr-7b-beta"
        ],
    )

if "messages" not in st.session_state.keys():
    st.session_state.messages = default_messages

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = default_messages
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

def answer_question(prompt):
    with st.spinner('Carregando documentos...'):
        docs = ai.load_documents(lang)
        db = ai.embed_documents(lang, docs)
        retriever = db.as_retriever()
    st.success('Documentos carregados!')

    with st.spinner('Carregando LLM...'):
        llm = ai.get_llm(repo_id)
    st.success('LLM carregada!')

    qa = ai.get_qa(llm, retriever)
    result = qa({'query': prompt})
    return result

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = answer_question(prompt)
            placeholder = st.empty()
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
