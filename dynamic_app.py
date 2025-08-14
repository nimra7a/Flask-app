import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import load_prompt
from langchain import LLMChain

hugging_face_key = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",
    api_key=hugging_face_key,
    task="text-generation",
    max_new_tokens=512
)

st.header("Research Tool")

paper_input = st.selectbox(
    "Select a research paper:",
    [
        "DIRS-CLAHS: Underwater Image Enhancement",
        "WGAN-GP for Alzheimer's MRI Classification",
        "Deep Q-Network for LunarLander",
        "ARIMA for Ethereum Price Forecasting"
    ]
)

style_input = st.selectbox(
    "Select the explanation style:",
    ["Beginner Friendly", "Code-Oriented", "Technical", "Mathematical"]
)

length_input = st.selectbox(
    "Select summary length:",
    ["Short (1-2 sentences)", "Medium (1-2 paragraphs)", "Long (detailed)"]
)

template = load_prompt('./template.json')

if st.button("Submit"):
    chain = LLMChain(prompt=template, llm=llm)
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result["text"])  # HuggingFaceEndpoint returns a dict
