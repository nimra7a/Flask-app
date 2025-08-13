import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate, load_prompt
import os



hugging_face_key = st.secrets["HUGGINGFACE_API_KEY"]
llm = HuggingFaceEndpoint(
  repo_id="mistralai/Mistral-7B-Instruct-v0.2",
  api_key=hugging_face_key,
  task = "text-generation"
)


st.header("Debug Research Tool")

# Debug secret
try:
    hugging_face_key = st.secrets["HUGGINGFACE_API_KEY"]
    st.write("Hugging Face key loaded ✅")
except Exception as e:
    st.error(f"Failed to load secret: {e}")

# Test LLM instantiation
try:
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        api_key=hugging_face_key,
        task="text-generation"
    )
    model = ChatHuggingFace(llm=llm)
    st.write("LLM initialized ✅")
except Exception as e:
    st.error(f"LLM init failed: {e}")

# Load prompt template
try:
    template = load_prompt("./template.json")
    st.write("Template loaded ✅")
except Exception as e:
    st.error(f"Template load failed: {e}")


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
    "Methametical"]
)

length_input = st.selectbox(
    "Select summary length:",
    ["Short (1-2 sentences)", "Medium (1-2 paragraphs)", "Long (detailed)"]
)

template = load_prompt('./template.json')


if st.button("Submit"):
  chain = template | model
  result = chain.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
  })
  st.write(result.content)