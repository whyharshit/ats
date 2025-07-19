from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()  
chatbot = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",temperature=0.5
   )
response = chatbot.invoke([HumanMessage(content="What is capital of India?")])
print(response.content)
