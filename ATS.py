import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import pdf2image
import io
import base64
from PIL import Image
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ Get API key and Poppler path from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
POPPLER_PATH = os.getenv("POPPLER_PATH", r"C:\poppler-24.08.0\Library\bin")

# ✅ Validate environment variables
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

# ✅ Load and process the uploaded PDF
def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        images = pdf2image.convert_from_bytes(uploaded_file.read(), poppler_path=POPPLER_PATH)
        first_page = images[0]

        # Convert first page to base64 JPEG
        image_byte_arr = io.BytesIO()
        first_page.save(image_byte_arr, format='JPEG')
        image_byte_arr = image_byte_arr.getvalue()

        # Return base64-encoded image string
        base64_image = base64.b64encode(image_byte_arr).decode()
        return base64_image
    else:
        raise FileNotFoundError("No file uploaded")


# ✅ Send prompt + image + job description to Gemini
def get_gemini_response(job_desc, base64_image, prompt):
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest")

    message = {
        "role": "user",
        "content": [
            job_desc,
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            },
            prompt
        ]
    }

    response = model.invoke([message])
    return response.content


# ✅ Streamlit UI
st.set_page_config(page_title="ATS Expert")
st.header("📄 ATS Tracking System")

input_text = st.text_area("🧾 Job Description:", key="input")
uploaded_file = st.file_uploader("📎 Upload your Resume (PDF only)", type=["pdf"])

if uploaded_file is not None:
    st.success("✅ PDF Uploaded Successfully")

submit1 = st.button("📌 Tell Me About the Resume")
submit3 = st.button("📊 Percentage Match with Job Description")

# ✅ Prompt Templates
input_prompt1 = """
You are an experienced Technical Human Resource Manager. Your task is to review the provided resume against the job description. 
Please share your professional evaluation on whether the candidate's profile aligns with the role. 
Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
"""

input_prompt3 = """
You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality. 
Your task is to evaluate the resume against the provided job description. 
Give me the percentage match, then list missing keywords, and finally provide your concluding thoughts.
"""

# ✅ Button actions
if submit1 or submit3:
    if uploaded_file is not None and input_text.strip():
        base64_image = input_pdf_setup(uploaded_file)
        selected_prompt = input_prompt1 if submit1 else input_prompt3
        response = get_gemini_response(input_text, base64_image, selected_prompt)

        st.subheader("🧠 Analysis:")
        st.write(response)
    else:
        st.warning("⚠️ Please upload a resume and enter the job description.")
