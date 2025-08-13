import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain import LLMChain
import os



hugging_face_key = st.secrets["HUGGINGFACE_API_KEY"]
llm = HuggingFaceEndpoint(
  repo_id="mistralai/Mistral-7B-Instruct-v0.2",
  api_key=hugging_face_key,
  task = "text-generation"
)

model = ChatHuggingFace(llm=llm)


st.header("Research Tool")

paper_input = st.selectbox(
    "Select a research paper:",
    ["DIRS-CLAHS: Underwater Image Enhancement",
    "WGAN-GP for Alzheimer's MRI Classification",
    "Deep Q-Network for LunarLander",
    "ARIMA for Ethereum Price Forecasting"
    ]
)

style_input = st.selectbox(
    "Select the explanation style:",
    ["Beginner Friendly", 
    "Code-Oriented", 
    "Technical", 
    "Methematical"]
)

length_input = st.selectbox(
    "Select summary length:",
    ["Short (1-2 sentences)", "Medium (1-2 paragraphs)", "Long (detailed)"]
)

template = load_prompt('./template.json')


if st.button("Submit"):
   chain = LLMChain(prompt=template, llm=model)
   result = chain.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
   })
   st.write(result.content)