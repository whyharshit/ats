from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Any,Annotated

# Load environment variables
load_dotenv()

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-latest",
    temperature=0.5
)

# ✅ Define schema using Pydantic
class Review(BaseModel):
    key_themes: Annotated[list[str],"write down all the key themes discussed in the review in a list"]
    summary: Annotated[str,"breif summary of the review"] #summary: str
    sentiment: str
    Battery: Any  # You can make this more detailed like another Pydantic model
    Camera: str

# Structured output
structured_model = model.with_structured_output(Review)

# Invoke the model
response = structured_model.invoke([
    HumanMessage(content="""I’ve been using the Redmi Note 13 Pro for the past month, and I must say it's a fantastic phone for its price. The 120Hz AMOLED display is buttery smooth and makes everyday usage feel premium. The 108MP camera is surprisingly good, especially in daylight, capturing details that rival even some flagship phones. Battery life is another strong point—thanks to the 5000mAh battery, it easily lasts me more than a day on moderate use. The Snapdragon 7s Gen 2 processor ensures a lag-free experience even when multitasking or playing games like BGMI. Plus, the 67W fast charging is a game-changer—I go from 0 to 100% in about 45 minutes! Overall, it's an excellent all-rounder for budget-conscious users looking for performance, display, and camera in one package.""")
])

# Print the structured output
print(response)
