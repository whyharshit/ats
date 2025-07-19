from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API key from .env
load_dotenv()

# Initialize the chatbot (Gemini model)
chatbot = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest")
prompt1 = PromptTemplate(template="Generate a detailed report on the {topic} ",input_variables=['topic'])

prompt2= PromptTemplate(template='Generate a 5 pointer summary form the following text \n {text}',
input_variables =['text'])

parser = StrOutputParser()



chain = prompt1 | chatbot | parser | prompt2 | chatbot | parser
result = chain.invoke({"topic": "cricket"})
print(result)
chain.get_graph().print_ascii()