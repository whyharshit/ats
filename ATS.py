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
            # For cloud environments, we'll use text extraction
            if is_cloud_environment():
                st.info("📄 Processing PDF as text in cloud environment...")
                # Extract text directly from PDF
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
                
                if not text_content.strip():
                    st.warning("⚠️ No text could be extracted from the PDF. This might be an image-based PDF.")
                    return None
                
                st.success(f"✅ Successfully extracted {len(text_content)} characters from PDF")
                return {"type": "text", "content": text_content}
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
                return {"type": "image", "content": base64_image}
        except Exception as e:
            st.error(f"❌ Error processing PDF: {str(e)}")
            st.info("💡 Try uploading a different PDF or ensure it's not password-protected.")
            return None
    else:
        raise FileNotFoundError("No file uploaded")


# ✅ Send prompt + image/text + job description to Gemini
def get_gemini_response(job_desc, pdf_data, prompt):
    try:
        model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest")

        # Handle the new data structure
        if isinstance(pdf_data, dict):
            if pdf_data["type"] == "text":
                # Text content from PDF
                message = {
                    "role": "user",
                    "content": [
                        f"Job Description: {job_desc}",
                        f"Resume Content: {pdf_data['content']}",
                        f"Analysis Request: {prompt}"
                    ]
                }
            elif pdf_data["type"] == "image":
                # Base64 image
                message = {
                    "role": "user",
                    "content": [
                        job_desc,
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{pdf_data['content']}"
                            }
                        },
                        prompt
                    ]
                }
        else:
            # Fallback for old format
            if isinstance(pdf_data, str) and len(pdf_data) > 1000:
                # Assume it's text content
                message = {
                    "role": "user",
                    "content": [
                        f"Job Description: {job_desc}",
                        f"Resume Content: {pdf_data}",
                        f"Analysis Request: {prompt}"
                    ]
                }
            else:
                # Assume it's base64 image
                message = {
                    "role": "user",
                    "content": [
                        job_desc,
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{pdf_data}"
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

input_text = st.text_area("🧾 Job Description:", key="input", 
                         placeholder="Paste any job description here - detailed or brief. The app will adapt its analysis accordingly.")
uploaded_file = st.file_uploader("📎 Upload your Resume (PDF only)", type=["pdf"])

if uploaded_file is not None:
    st.success("✅ PDF Uploaded Successfully")

# Add info about adaptive analysis
with st.expander("ℹ️ How the Analysis Works"):
    st.markdown("""
    **Adaptive Analysis System:**
    
    🎯 **Detailed Job Descriptions:** Full comprehensive analysis using all available criteria
    
    📝 **Brief Job Descriptions:** Extracts key requirements and makes reasonable assumptions based on job title and industry standards
    
    🔍 **Vague Job Descriptions:** Focuses on transferable skills and general role expectations
    
    The app automatically adapts its analysis based on the level of detail in your job description!
    """)

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
You are an experienced Technical Human Resource Manager with expertise in talent acquisition and candidate evaluation. Your task is to provide a comprehensive, honest assessment of the candidate's resume against the job description, regardless of how detailed or brief the job description is.

**ADAPTIVE EVALUATION APPROACH:**
- **For detailed job descriptions:** Use all available criteria for thorough assessment
- **For brief job descriptions:** Extract key requirements and make reasonable assumptions based on job title and industry standards
- **For vague job descriptions:** Focus on transferable skills and general role expectations

**EVALUATION CRITERIA:**
1. **Role Alignment** - How well does the candidate's background match the job requirements (explicit and implicit)?
2. **Technical Competency** - Does the candidate have the necessary technical skills for the role?
3. **Experience Relevance** - Is their work experience applicable to this position?
4. **Education & Certifications** - Do they meet the educational requirements?
5. **Communication & Presentation** - How well is their experience communicated in the resume?

**ANALYSIS REQUIREMENTS:**
- **First, analyze the job description:** Extract requirements and note any assumptions made
- **Provide a candid assessment** of fit (don't sugar-coat)
- **Identify specific strengths** that make them suitable for the role
- **Highlight critical gaps** that could be deal-breakers
- **Rate overall suitability** (Poor/Fair/Good/Excellent) with reasoning
- **Provide actionable feedback** for improvement

**RESPONSE FORMAT:**
- **Job Description Summary:** [Brief analysis of what requirements were identified]
- **Overall Assessment:** [Poor/Fair/Good/Excellent] with reasoning
- **Key Strengths:** [specific points that make them suitable]
- **Critical Gaps:** [specific concerns or missing requirements]
- **Recommendation:** [Hire/Consider/Reject with detailed reasoning]
- **Improvement Suggestions:** [specific, actionable advice]

**IMPORTANT:** Be direct and honest. If the job description is too vague, acknowledge this and provide a general assessment. If the candidate is not a good fit, explain why clearly.
"""

input_prompt3 = """
You are an expert ATS (Applicant Tracking System) scanner and HR professional. Your task is to provide a detailed, accurate analysis of the resume against the job description, regardless of how detailed or brief the job description is.

**ADAPTIVE ANALYSIS APPROACH:**
- If job description is detailed: Use comprehensive scoring methodology
- If job description is brief: Extract key requirements and make reasonable assumptions based on job title and industry standards
- If job description is vague: Focus on transferable skills and general role requirements

**SCORING METHODOLOGY (Adaptive):**
- **Technical Skills: 40%** - Programming languages, tools, software, technical competencies
- **Experience & Education: 30%** - Relevant work history, degrees, certifications, years of experience
- **Soft Skills: 20%** - Communication, leadership, teamwork, problem-solving
- **Role Alignment: 10%** - How well the candidate's background fits the role type

**ANALYSIS REQUIREMENTS:**
1. **First, analyze the job description:**
   - Extract explicit requirements
   - Identify implicit requirements based on job title
   - Note any industry-specific expectations

2. **Then, calculate a precise percentage match** (0-100%) based on available information

3. **Provide detailed breakdown:**
   - Score each category with reasoning
   - If job description lacks detail, explain your assumptions

4. **List missing keywords/skills** (both explicit and implicit)

5. **Identify strengths and weaknesses** with specific examples

6. **Provide actionable recommendations** for improvement

**RESPONSE FORMAT:**
- **Job Description Analysis:** [Brief summary of what you extracted]
- **Overall Match: X%**
- **Breakdown:** 
  - Technical Skills: X% (reasoning)
  - Experience & Education: X% (reasoning)
  - Soft Skills: X% (reasoning)
  - Role Alignment: X% (reasoning)
- **Missing Keywords/Skills:** [list with context]
- **Strengths:** [specific points]
- **Weaknesses:** [specific areas for improvement]
- **Recommendations:** [actionable advice]

**IMPORTANT:** Be honest and critical. Don't inflate scores. If the job description is too vague, say so and provide a general assessment.
"""

# ✅ Button actions
if submit1 or submit3:
    if uploaded_file is not None and input_text.strip():
        pdf_data = input_pdf_setup(uploaded_file)
        if pdf_data is not None:
            # Show preview of extracted content
            if pdf_data["type"] == "text":
                with st.expander("📄 Preview Extracted Text"):
                    st.text(pdf_data["content"][:500] + "..." if len(pdf_data["content"]) > 500 else pdf_data["content"])
            
            selected_prompt = input_prompt1 if submit1 else input_prompt3
            
            # Show analysis parameters for percentage match
            if submit3:
                st.info("📊 **Analysis Parameters:** Technical Skills (40%) + Experience (30%) + Soft Skills (20%) + Project Alignment (10%)")
            
            response = get_gemini_response(input_text, pdf_data, selected_prompt)

            st.subheader("🧠 Analysis:")
            st.write(response)
        else:
            st.error("❌ Failed to process the PDF. Please try a different file.")
    else:
        st.warning("⚠️ Please upload a resume and enter the job description.")
