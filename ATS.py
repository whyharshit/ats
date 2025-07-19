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
# For cloud deployment, Poppler is installed system-wide
POPPLER_PATH = os.getenv("POPPLER_PATH", None)

# ✅ Validate environment variables
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found in environment variables.")
    st.markdown("""
    ### 🔧 How to fix this:
    
    **For Streamlit Cloud:**
    1. Go to your app settings at [share.streamlit.io](https://share.streamlit.io)
    2. Click on your app → Settings → Secrets
    3. Add: `GOOGLE_API_KEY = "your_api_key_here"`
    
    **For local development:**
    1. Create a `.env` file in your project directory
    2. Add: `GOOGLE_API_KEY=your_api_key_here`
    
    **Get your API key from:** [Google AI Studio](https://makersuite.google.com/app/apikey)
    """)
    st.stop()

# ✅ For cloud deployment, we'll need to handle PDF processing differently
def is_cloud_environment():
    """Check if running in a cloud environment"""
    return os.getenv("STREAMLIT_SERVER_PORT") is not None or os.getenv("PORT") is not None

# ✅ Test API key functionality
def test_api_key():
    """Test if the API key is working"""
    try:
        model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest")
        response = model.invoke("Hello, this is a test message.")
        return True, "API key is working correctly!"
    except Exception as e:
        return False, f"API key test failed: {str(e)}"

# ✅ Load and process the uploaded PDF
def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        try:
            # For cloud environments, we'll use a different approach
            if is_cloud_environment():
                st.warning("⚠️ PDF processing in cloud environments may have limitations. Please ensure your PDF is clear and readable.")
                # For now, we'll extract text directly from PDF
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text()
                return text_content
            else:
                # Local environment with Poppler
                images = pdf2image.convert_from_bytes(uploaded_file.read(), poppler_path=POPPLER_PATH)
                first_page = images[0]

                # Convert first page to base64 JPEG
                image_byte_arr = io.BytesIO()
                first_page.save(image_byte_arr, format='JPEG')
                image_byte_arr = image_byte_arr.getvalue()

                # Return base64-encoded image string
                base64_image = base64.b64encode(image_byte_arr).decode()
                return base64_image
        except Exception as e:
            st.error(f"❌ Error processing PDF: {str(e)}")
            st.info("💡 Try uploading a different PDF or ensure it's not password-protected.")
            return None
    else:
        raise FileNotFoundError("No file uploaded")


# ✅ Send prompt + image/text + job description to Gemini
def get_gemini_response(job_desc, pdf_content, prompt):
    try:
        model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest")

        # Check if pdf_content is base64 image or text
        if isinstance(pdf_content, str) and (pdf_content.startswith('data:image') or len(pdf_content) > 1000):
            # It's likely text content from PDF
            message = {
                "role": "user",
                "content": [
                    f"Job Description: {job_desc}",
                    f"Resume Content: {pdf_content}",
                    f"Analysis Request: {prompt}"
                ]
            }
        else:
            # It's base64 image
            message = {
                "role": "user",
                "content": [
                    job_desc,
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{pdf_content}"
                        }
                    },
                    prompt
                ]
            }

        response = model.invoke([message])
        return response.content
        
    except Exception as e:
        error_msg = str(e)
        if "ResourceExhausted" in error_msg:
            return """
            ❌ **API Quota Exceeded**
            
            Your Google API key has reached its quota limit. Here's how to fix this:
            
            **Option 1: Check Your Quota**
            1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. Check your usage and quotas
            3. Wait for quota reset or upgrade your plan
            
            **Option 2: Generate New API Key**
            1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. Create a new API key
            3. Update it in Streamlit Cloud secrets
            
            **Option 3: Use Local Development**
            Run the app locally with your own API key for unlimited usage.
            """
        elif "InvalidArgument" in error_msg or "PermissionDenied" in error_msg:
            return """
            ❌ **API Key Error**
            
            Your Google API key appears to be invalid or doesn't have proper permissions.
            
            **How to fix:**
            1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
            2. Generate a new API key
            3. Update it in Streamlit Cloud secrets
            """
        else:
            return f"""
            ❌ **Unexpected Error**
            
            An error occurred while processing your request: {error_msg}
            
            **Troubleshooting:**
            1. Check your internet connection
            2. Verify your API key is correct
            3. Try uploading a smaller PDF file
            """


# ✅ Streamlit UI
st.set_page_config(page_title="ATS Expert")
st.header("📄 ATS Tracking System")

input_text = st.text_area("🧾 Job Description:", key="input")
uploaded_file = st.file_uploader("📎 Upload your Resume (PDF only)", type=["pdf"])

if uploaded_file is not None:
    st.success("✅ PDF Uploaded Successfully")

submit1 = st.button("📌 Tell Me About the Resume")
submit3 = st.button("📊 Percentage Match with Job Description")

# Add API key test button
if st.button("🔑 Test API Key"):
    success, message = test_api_key()
    if success:
        st.success(message)
    else:
        st.error(message)

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
