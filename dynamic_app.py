import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain import LLMChain

hugging_face_key = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# Use a free model
llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",
    api_key=hugging_face_key,
    task="text-generation",
    max_new_tokens=512
)

# Define the prompt template
template = """
Summarize the research paper titled "{paper_input}" in a {style_input} style.
Provide a {length_input} summary.
"""
prompt = PromptTemplate(
    input_variables=["paper_input", "style_input", "length_input"],
    template=template
)

chain = LLMChain(prompt=prompt, llm=llm)

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

if st.button("Submit"):
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result["text"])  # HuggingFaceEndpoint returns dict with 'text'
