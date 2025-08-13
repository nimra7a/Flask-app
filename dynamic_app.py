import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain import LLMChain

# Load Hugging Face key from secrets
hf_key = st.secrets["HUGGINGFACE_API_KEY"]

# Initialize Hugging Face LLM
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    api_key=hf_key,
    task="chat-completion"  # Must match model type
)

model = ChatHuggingFace(llm=llm)

# Streamlit UI
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

# Prompt template
template = PromptTemplate(
    input_variables=["paper_input", "style_input", "length_input"],
    template="""
Summarize the research paper titled "{paper_input}".
Explanation style: {style_input}
Summary length: {length_input}
"""
)

# Run LLMChain on button click
if st.button("Submit"):
    chain = LLMChain(prompt=template, llm=model)
    result = chain.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input
    })
    st.write(result.content)
