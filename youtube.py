import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import re
import os
import requests
import time
import random

# Load environment variables
load_dotenv()

# Get API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validate environment variables
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

st.set_page_config(page_title="YouTube Q and A", layout="centered")
st.title("YouTube Transcript Question Answering App")
st.write("Enter a YouTube video link and ask a question based on the transcript.")

# Sidebar for advanced options
with st.sidebar:
    st.header("⚙️ Advanced Options")
    use_proxy = st.checkbox("Use Proxy (if available)", value=False, help="Enable proxy to avoid IP blocking")
    retry_attempts = st.slider("Retry Attempts", min_value=1, max_value=5, value=3, help="Number of retry attempts if transcript fails")
    
    if use_proxy:
        st.info("💡 Proxy feature helps avoid YouTube IP blocking. Free proxies may be unreliable.")

# Main input section
input_method = st.radio(
    "Choose input method:",
    ["YouTube URL", "Manual Transcript"],
    help="Select how you want to provide the video content"
)

video_url = None
manual_transcript = None

if input_method == "YouTube URL":
    video_url = st.text_input("Enter YouTube Video URL")
else:
    manual_transcript = st.text_area(
        "Paste the video transcript here:",
        height=200,
        help="Copy the transcript from YouTube's caption/transcript feature and paste it here"
    )
    
    if st.button("📋 How to get transcript"):
        st.markdown("""
        ### 📋 How to get YouTube transcript:
        
        1. **Go to the YouTube video**
        2. **Click the "..." (three dots) below the video**
        3. **Select "Show transcript"**
        4. **Copy all the text from the transcript panel**
        5. **Paste it in the text area above**
        
        **Note:** This method works even when the API is blocked!
        """)

question = st.text_input("Enter your question")

def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

def get_free_proxies():
    """Get a list of free proxies (use with caution)"""
    try:
        response = requests.get('https://api.proxyscrape.com/v2/?request=get&protocol=http&timeout=10000&country=all&ssl=all&anonymity=all')
        if response.status_code == 200:
            proxies = response.text.strip().split('\n')
            return [f"http://{proxy}" for proxy in proxies if proxy]
    except:
        pass
    return []

def get_transcript_with_retry(video_id, max_attempts=3, use_proxy=False):
    """Get transcript with retry logic and optional proxy support"""
    proxies = []
    if use_proxy:
        proxies = get_free_proxies()
    
    for attempt in range(max_attempts):
        try:
            if use_proxy and proxies:
                # Try with a random proxy
                proxy = random.choice(proxies)
                st.info(f"🔄 Attempt {attempt + 1}: Trying with proxy...")
                
                # Configure proxy for requests
                import os
                os.environ['HTTP_PROXY'] = proxy
                os.environ['HTTPS_PROXY'] = proxy
                
                # Add delay to avoid rate limiting
                time.sleep(random.uniform(1, 3))
            
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            return transcript_list
            
        except TranscriptsDisabled:
            raise TranscriptsDisabled("Transcript is disabled for this video.")
        except Exception as e:
            error_msg = str(e)
            if "blocked" in error_msg.lower() or "ip" in error_msg.lower():
                if attempt < max_attempts - 1:
                    st.warning(f"⚠️ IP blocked on attempt {attempt + 1}. Retrying...")
                    time.sleep(random.uniform(2, 5))  # Wait before retry
                    continue
                else:
                    raise Exception("YouTube is blocking requests from this IP. Try using a proxy or different network.")
            else:
                raise e
    
    raise Exception("Failed to retrieve transcript after multiple attempts.")

def process_transcript_and_answer(transcript_text, question):
    """Process transcript and generate answer using LangChain"""
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript_text])

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest")
        prompt = PromptTemplate(
            template="""
            You are a helpful assistant.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, just say you don't know.

            {context}
            Question: {question}
            """,
            input_variables=['context', 'question']
        )

        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })

        parser = StrOutputParser()
        chain = parallel_chain | prompt | llm | parser

        answer = chain.invoke(question)
        return answer
    except Exception as e:
        raise Exception(f"Error processing transcript: {str(e)}")

# Main processing logic
if (video_url or manual_transcript) and question:
    transcript_text = None
    
    if input_method == "YouTube URL":
        video_id = extract_video_id(video_url)
        
        if not video_id:
            st.error("Invalid YouTube URL. Please check and try again.")
        else:
            with st.spinner("Loading transcript and preparing response..."):
                try:
                    # Get transcript with retry logic
                    transcript_list = get_transcript_with_retry(video_id, retry_attempts, use_proxy)
                    transcript_text = " ".join(chunk["text"] for chunk in transcript_list)
                    
                except TranscriptsDisabled:
                    st.error("❌ Transcript is disabled for this video.")
                    st.info("💡 Try a different video that has captions enabled, or use the manual transcript option.")
                except Exception as e:
                    error_msg = str(e)
                    if "blocked" in error_msg.lower() or "ip" in error_msg.lower():
                        st.error("❌ YouTube IP Blocking Detected")
                        st.markdown("""
                        ### 🔧 Solutions to try:
                        
                        **1. Enable Proxy (Recommended)**
                        - Check the "Use Proxy" option in the sidebar
                        - This will attempt to use free proxies to bypass the block
                        
                        **2. Use Manual Transcript (Easiest)**
                        - Switch to "Manual Transcript" input method
                        - Copy transcript directly from YouTube
                        - This bypasses all API restrictions
                        
                        **3. Try Different Network**
                        - Use a different internet connection
                        - Try from a mobile hotspot
                        
                        **4. Wait and Retry**
                        - YouTube blocks are usually temporary
                        - Wait 10-15 minutes and try again
                        
                        **5. Alternative Solutions**
                        - Use a VPN service
                        - Try from a different location
                        """)
                    elif "quota" in error_msg.lower():
                        st.error("❌ API Quota Exceeded")
                        st.info("💡 Check your Google API key usage and quotas.")
                    else:
                        st.error(f"❌ An error occurred: {error_msg}")
                        st.info("💡 Please check your API key and try again.")
    else:
        # Manual transcript input
        if not manual_transcript.strip():
            st.error("Please paste a transcript in the text area.")
        else:
            transcript_text = manual_transcript
    
    # Process transcript and generate answer
    if transcript_text:
        with st.spinner("Processing transcript and generating answer..."):
            try:
                answer = process_transcript_and_answer(transcript_text, question)
                st.subheader("Answer")
                st.write(answer)
                
                # Show transcript preview
                with st.expander("📄 View Transcript Preview"):
                    st.text(transcript_text[:500] + "..." if len(transcript_text) > 500 else transcript_text)
                    
            except Exception as e:
                st.error(f"❌ Error processing transcript: {str(e)}")
                st.info("💡 Please check your API key and try again.")

# Add helpful information in the sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📚 Help")
    st.markdown("""
    **How to use:**
    1. Choose input method (URL or Manual)
    2. Provide video content
    3. Ask a question about the content
    4. Get AI-powered answers
    
    **Tips:**
    - Manual transcript bypasses all API restrictions
    - Videos must have English captions enabled
    - Longer videos may take more time to process
    - Use specific questions for better answers
    """)
    
    st.markdown("### 🛠️ Troubleshooting")
    if st.button("Show Common Issues"):
        st.markdown("""
        **IP Blocking:**
        - Enable proxy in advanced options
        - Use manual transcript option
        - Try different network
        - Wait 10-15 minutes
        
        **No Transcript:**
        - Check if video has captions
        - Try different video
        - Use manual transcript option
        - Some videos disable transcripts
        
        **API Errors:**
        - Check your Google API key
        - Verify quota limits
        - Ensure stable internet
        - Use manual transcript as fallback
        """)
