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

video_url = st.text_input("Enter YouTube Video URL")
question = st.text_input("Enter your question")

def extract_video_id(url):
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

if video_url and question:
    video_id = extract_video_id(video_url)

    if not video_id:
        st.error("Invalid YouTube URL. Please check and try again.")
    else:
        with st.spinner("Loading transcript and preparing response..."):
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
                transcript = " ".join(chunk["text"] for chunk in transcript_list)

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.create_documents([transcript])

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

                st.subheader("Answer")
                st.write(answer)

            except TranscriptsDisabled:
                st.error("Transcript is disabled for this video.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
