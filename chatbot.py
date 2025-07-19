from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

# Initialize the chatbot (Gemini model)
chatbot = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest")

# Initialize structured chat history (as message objects)
chat_history = []

while True:
    # Get user input
    user_input = input("You: ")

    # Check if user wants to exit
    if user_input.lower() == "bye":
        print("AI: Bye! Have a great day.")
        break

    # Append user input as a HumanMessage
    chat_history.append(HumanMessage(content=user_input))

    # Send message history to Gemini
    result = chatbot.invoke(chat_history)

    # Append AI's response as an AIMessage
    chat_history.append(AIMessage(content=result.content))

    # Print AI's response
    print("AI:", result.content)
