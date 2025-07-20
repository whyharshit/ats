import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import re
import os

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="YouTube Q&A", layout="centered")
st.title("🎥 YouTube Transcript Q&A")
st.write("Ask questions about any YouTube video with captions!")

# Check API key first
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found!")
    st.markdown("""
    ### 🔧 How to fix this:
    
    **1. Get your API key:**
    - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
    - Click "Create API Key"
    - Copy the key
    
    **2. Add to .env file:**
    Create a `.env` file in your project folder and add:
    ```
    GOOGLE_API_KEY=your_api_key_here
    ```
    
    **3. Restart the app:**
    Stop and run `streamlit run youtube.py` again
    """)
    st.stop()

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def format_docs(docs):
    """Format documents into context text"""
    return "\n\n".join(doc.page_content for doc in docs)

def test_api_key():
    """Test if the API key is valid"""
    try:
        # Test with a simple embedding
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embeddings.embed_query("test")
        return True, "API key is valid"
    except Exception as e:
        return False, str(e)

# Test API key on startup
api_valid, api_message = test_api_key()
if not api_valid:
    st.error("❌ Invalid API Key!")
    st.markdown(f"""
    **Error:** {api_message}
    
    ### 🔧 How to fix:
    
    **1. Check your API key:**
    - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
    - Make sure the key is active and not restricted
    
    **2. Update your .env file:**
    ```
    GOOGLE_API_KEY=your_new_api_key_here
    ```
    
    **3. Restart the app**
    """)
    st.stop()

# Input section
video_url = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
question = st.text_input("Ask a question about the video:")

if st.button("🚀 Get Answer", type="primary"):
    if video_url and question:
        with st.spinner("Processing..."):
            try:
                # Extract video ID
                video_id = extract_video_id(video_url)
                if not video_id:
                    st.error("❌ Invalid YouTube URL")
                    st.stop()
                
                # Get transcript
                st.info("📹 Fetching transcript...")
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
                transcript = " ".join(chunk["text"] for chunk in transcript_list)
                st.success(f"✅ Transcript loaded ({len(transcript)} characters)")
                
                # Process with AI
                st.info("🤖 Processing with AI...")
                
                # Split transcript
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript])
                
                # Create embeddings and vector store
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.from_documents(chunks, embeddings)
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
                
                # Get relevant context
                docs = retriever.get_relevant_documents(question)
                context = format_docs(docs)
                
                # Generate answer
                llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest")
                prompt = PromptTemplate(
                    template="""
                    You are a helpful assistant. Answer ONLY from the provided transcript context.
                    If the context is insufficient, just say you don't know.

                    Context: {context}
                    Question: {question}
                    """,
                    input_variables=['context', 'question']
                )
                
                formatted_prompt = prompt.format(context=context, question=question)
                response = llm.invoke(formatted_prompt)
                
                # Display answer
                st.subheader("💡 Answer:")
                st.write(response.content)
                
                # Show context used
                with st.expander("📄 Context used"):
                    st.text(context[:500] + "..." if len(context) > 500 else context)
                
            except TranscriptsDisabled:
                st.error("❌ No captions available for this video")
                st.info("💡 Try a different video with English captions enabled")
                
            except Exception as e:
                error_msg = str(e)
                if "blocked" in error_msg.lower() or "ip" in error_msg.lower():
                    st.error("❌ YouTube IP Blocking Detected")
                    st.markdown("""
                    **Solutions:**
                    - Try a different network
                    - Use a VPN
                    - Wait a few minutes and try again
                    """)
                elif "API_KEY_INVALID" in error_msg or "400" in error_msg:
                    st.error("❌ API Key Error")
                    st.markdown("""
                    **Your API key is invalid or expired.**
                    
                    **How to fix:**
                    1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
                    2. Create a new API key
                    3. Update your `.env` file
                    4. Restart the app
                    """)
                else:
                    st.error(f"❌ Error: {error_msg}")
                    st.info("💡 Please check your Google API key in .env file")
    else:
        st.warning("⚠️ Please enter both URL and question")

# Sidebar with help
with st.sidebar:
    st.header("ℹ️ Help")
    st.markdown("""
    **How to use:**
    1. Paste a YouTube URL
    2. Ask a question about the video
    3. Get AI-powered answers!
    
    **Requirements:**
    - Video must have English captions
    - Valid Google API key in .env file
    
    **Tips:**
    - Ask specific questions for better answers
    - Works with any YouTube video with captions
    """)
    
    if st.button("🔑 Check API Key"):
        if api_valid:
            st.success("✅ API key is valid")
            st.info(f"Key: {GOOGLE_API_KEY[:10]}...")
        else:
            st.error("❌ API key is invalid")
            st.info("Please update your .env file with a valid key")