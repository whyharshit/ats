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

st.set_page_config(page_title="YouTube Q&A", layout="centered")
st.title("🎥 YouTube Transcript Q&A")
st.write("Ask questions about any YouTube video with captions!")

def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

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
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
                transcript = " ".join(chunk["text"] for chunk in transcript_list)
                # Split transcript
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript])
                # Embeddings and vector store
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
                st.error("No captions available for this video")
                st.info("Try a different video with English captions enabled")
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Too Many Requests" in error_msg:
                    st.error("YouTube Rate Limit Hit — Please wait a few minutes.")
                elif "403" in error_msg or "forbidden" in error_msg.lower():
                    st.error("Access forbidden. Check your network or try a VPN.")
                elif "TranscriptsDisabled" in error_msg:
                    st.error("Captions not available for this video.")
           
                elif "API_KEY_INVALID" in error_msg or "400" in error_msg:
                    st.error("API Key Error")
                    
                    st.markdown("""
                    **Your API key is invalid or expired.**
                    
                    **How to fix:**
                    1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
                    2. Create a new API key
                    3. Update your `.env` file
                    4. Restart the app
                    """)
                else:
                    st.error(f"Error: {error_msg}")
                    st.info("Please check your Google API key in .env file")
    else:
        st.warning("⚠️ Please enter both URL and question")


