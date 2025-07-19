from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import streamlit as st
from dotenv import load_dotenv
load_dotenv() 


chatbot = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",temperature=0.5
   )

st.header('Research Tool') 
user_input = st.text_input("Enter your prompt")
if st.button('Summarize'):
    
  response = chatbot.invoke([HumanMessage(content=user_input)])
  st.write(response.content)

