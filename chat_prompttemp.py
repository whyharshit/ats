# Import the Gemini Chat model from LangChain's Google GenAI integration
from langchain_google_genai import ChatGoogleGenerativeAI

# Import tools to build custom prompt templates and manage message formats
from langchain_core.prompts import (
    ChatPromptTemplate,                # For building chat prompts with structure
    SystemMessagePromptTemplate,       # Template for system instructions
    HumanMessagePromptTemplate,        # Template for user's messages
    MessagesPlaceholder                # Placeholder to include dynamic chat history
)

# Import message classes to represent actual messages in conversation
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Import dotenv to securely load the API key from the `.env` file
from dotenv import load_dotenv

# ✅ Load your Google API Key and other env variables from `.env` file
load_dotenv()

# ✅ Create a chatbot instance using Gemini 1.5 Flash model
chatbot = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest")

# ✅ Define how the prompt should be structured for the chatbot
# We're giving:
# - A system instruction saying it's a domain expert
# - A placeholder for past chat history (optional but useful for continuity)
# - A new human question asking to explain a topic simply
chat_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful {domain} expert."),   # dynamic domain (e.g. cricket expert)
    MessagesPlaceholder(variable_name='chat_history'),                                 # history will be inserted here
    HumanMessagePromptTemplate.from_template("Explain in simple terms, what is {topic}?")  # user query
])

# ✅ Create an empty list to store past messages
# This history helps Gemini remember what has been said before
chat_history = []

# ✅ Try to load chat history from a file (if it exists)
# We use this so that the model can remember previous exchanges
try:
    with open('chat_history.txt') as f:
        for line in f:
            # If the line starts with "You:", treat it as a user's message
            if line.startswith("You: "):
                chat_history.append(HumanMessage(content=line.replace("You: ", "").strip()))
            # If the line starts with "AI:", treat it as AI's response
            elif line.startswith("AI: "):
                chat_history.append(AIMessage(content=line.replace("AI: ", "").strip()))
except FileNotFoundError:
    # If the file is not found, just use an empty history
    print("chat_history.txt not found. Continuing with empty history.")

# ✅ Now we fill in the placeholders in the prompt
# We're telling Gemini:
# - The domain is 'cricket'
# - The topic is 'stump'
# - The chat_history (if any) is also provided
prompt = chat_template.invoke({
    'domain': 'cricket',
    'chat_history': chat_history,
    'topic': 'stump'
})

# ✅ Send the final structured prompt to the Gemini model and get the answer
result = chatbot.invoke(prompt)

# ✅ Print the chatbot’s reply to the console
print("AI:", result.content)

# ✅ Save the new exchange (Q&A) into the history file for continuity in future
# This allows future runs of the script to remember past chats
with open('chat_history.txt', 'a') as f:
    f.write(f"You: Explain in simple terms, what is stump?\n")
    f.write(f"AI: {result.content}\n")
