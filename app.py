import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
import secret

os.environ["HUGGINGFACEHUB_API_TOKEN"] = secret.HuggingFaceHub_ACCESS_TOKEN
llm = HuggingFaceEndpoint(
  repo_id="mistralai/Mistral-7B-Instruct-v0.2",
  task = "text-generation"
)

model = ChatHuggingFace(llm=llm)


st.header("Research Tool")
user_input = st.text_input("Enter your research question")

if st.button("Submit"):
  result = model.invoke(user_input)
  st.write(result.content)