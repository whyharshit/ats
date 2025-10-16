import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pytube import YouTube
import re
import os

# Load environment variables
load_dotenv()

st.set_page_config(page_title="YouTube Q&A", layout="centered")
st.title("🎥 YouTube Transcript Q&A")
st.write("Ask questions about any YouTube video with captions!")

# Extract YouTube video ID
def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

# Format docs for context display
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Hybrid transcript fetcher
def get_youtube_transcript(video_id):
    transcript = ""

    try:
        # 🔹 Primary method
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript

    except Exception as e1:
        # 🔹 Fallback 1: using list_transcripts
        try:
            transcript_obj = YouTubeTranscriptApi.list_transcripts(video_id).find_transcript(['en'])
            transcript_list = transcript_obj.fetch()
            transcript = " ".join(t["text"] for t in transcript_list)
            return transcript

        except Exception as e2:
            # 🔹 Fallback 2: pytube captions
            try:
                yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
                caption = yt.captions.get_by_language_code('en')
                if caption:
                    transcript = caption.generate_srt_captions()
                    return transcript
                else:
                    raise Exception("No English captions found via pytube.")
            except Exception as e3:
                raise Exception(f"Transcript unavailable. Errors:\n1️⃣ {e1}\n2️⃣ {e2}\n3️⃣ {e3}")

# --- Streamlit App Logic ---
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

                # Get transcript (smart fallback)
                transcript = get_youtube_transcript(video_id)
                if not transcript.strip():
                    st.error("No transcript found for this video.")
                    st.stop()

                # Split transcript
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript])

                # Embeddings + Vector DB
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                vector_store = FAISS.from_documents(chunks, embeddings)
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

                # Retrieve context
                docs = retriever.get_relevant_documents(question)
                context = format_docs(docs)

                # Generate Answer using Gemini
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

                # Display results
                st.subheader("💡 Answer:")
                st.write(response.content)

                # Context preview
                with st.expander("📄 Context used"):
                    st.text(context[:800] + "..." if len(context) > 800 else context)

            except TranscriptsDisabled:
                st.error("❌ Captions are disabled for this video.")
                st.info("Try a different video with English captions enabled.")
            except Exception as e:
                error_msg = str(e)
                st.error("⚠️ Something went wrong.")
                st.code(error_msg)

                # Helpful hints
                if "quota" in error_msg.lower() or "403" in error_msg:
                    st.warning("⚠️ API quota exceeded or forbidden. Try later or use another API key.")
                elif "API_KEY_INVALID" in error_msg or "400" in error_msg:
                    st.warning("❌ Invalid API key. Please regenerate in Google AI Studio and update your `.env` file.")
                elif "No English captions" in error_msg:
                    st.info("Try a video with English captions or manually uploaded subtitles.")
                else:
                    st.info("Please recheck your internet connection or video link.")
    else:
        st.warning("⚠️ Please enter both URL and question.")
