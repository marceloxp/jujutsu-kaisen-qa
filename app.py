import time
import streamlit as st
import pandas as pd
import numpy as np
import multilang
import ai

import gc
gc.collect()

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
    default_messages = [{"role": "assistant", "content": multilang.get_translation(lang, 'hi')}]

    repo_id = st.selectbox(
        "Select Model",
        [
            "t5-small",
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
    with st.spinner('Processing documents...'):
        docs = ai.load_documents(lang)
        db = ai.embed_documents(lang, docs)
        retriever = db.as_retriever()
    st.toast('Documents processed!', icon='🎉')

    with st.spinner('Loading LLM...'):
        qa = ai.get_qa_v2(repo_id, retriever)
    st.toast('LLM loaded!', icon='🎉')

    with st.spinner('Making a query...'):
        result = qa({'query': prompt})
    st.toast('Query completed!', icon='🎉')
    return result['result']

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if prompt == '/help':
                response = 'Language: ' + language
            else:
                response = answer_question(prompt)
            st.write(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
