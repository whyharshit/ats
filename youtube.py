import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import re
import os

# Load environment variables
load_dotenv()

st.set_page_config(page_title="YouTube Q and A", layout="centered")
st.title("YouTube Transcript Question Answering App")
st.write("Enter a YouTube video link and ask a question based on the transcript.")

# Input method selection
input_method = st.radio("Choose input method:", ["YouTube URL", "Manual Transcript"])

video_url = None
manual_transcript = None

if input_method == "YouTube URL":
    video_url = st.text_input("Enter YouTube Video URL")
else:
    manual_transcript = st.text_area("Paste the video transcript here:", height=200)
    if st.button("📋 How to get transcript"):
        st.markdown("""
        **How to get YouTube transcript:**
        1. Go to YouTube video
        2. Click "..." (three dots) below video
        3. Select "Show transcript"
        4. Copy all text and paste above
        """)

question = st.text_input("Enter your question")

def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

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
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
                    transcript_text = " ".join(chunk["text"] for chunk in transcript_list)
                    
                except TranscriptsDisabled:
                    st.error("❌ Transcript is disabled for this video.")
                    st.info("💡 Try using 'Manual Transcript' option instead.")
                except Exception as e:
                    error_msg = str(e)
                    if "blocked" in error_msg.lower() or "ip" in error_msg.lower():
                        st.error("❌ YouTube IP Blocking Detected")
                        st.markdown("""
                        **Quick Fix:**
                        1. Switch to "Manual Transcript" input method
                        2. Copy transcript from YouTube manually
                        3. Paste and ask your question
                        
                        This bypasses all API restrictions!
                        """)
                    else:
                        st.error(f"❌ An error occurred: {error_msg}")
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
