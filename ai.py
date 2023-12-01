import os
import streamlit as st
import textwrap
import random
import pandas as pd
from langchain import HuggingFaceHub
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"
import torch
torch.cuda.empty_cache()

def load_documents(lang):
    loader = TextLoader("./sample_data/data-"+lang+".txt", encoding="utf-8")
    raw_documents = loader.load()
    text_splitter = CharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
    )
    docs = text_splitter.split_documents(raw_documents)
    return docs

def embed_documents(lang, docs):
    model_sentence = "jmbrito/ptbr-similarity-e5-small" if lang == "pt" else "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_sentence)
    db = Chroma.from_documents(docs, embeddings)
    return db

def get_qa(repo_id, retriever):
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={'temperature': 0.5, 'max_new_tokens': 1024})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, return_source_documents=False)
    return qa

def get_qa_v2(repo_id, retriever):
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo_id)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    llm = HuggingFacePipeline(
        pipeline = pipe,
        model_kwargs={"temperature": 0.5, "max_length": 1024},
    )
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, return_source_documents=False)
    return qa