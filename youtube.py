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

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def format_docs(retrieved_docs):
    """Format retrieved documents into context text"""
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

def main():
    print("🎥 YouTube Transcript Q&A Tool")
    print("=" * 40)
    
    # Get YouTube URL or video ID
    video_input = input("Enter YouTube URL or Video ID: ").strip()
    
    # Extract video ID if URL provided
    if "youtube.com" in video_input or "youtu.be" in video_input:
        video_id = extract_video_id(video_input)
        if not video_id:
            print("❌ Invalid YouTube URL. Please check and try again.")
            return
    else:
        video_id = video_input
    
    print(f"📹 Processing video ID: {video_id}")
    
    try:
        # Get transcript
        print("🔄 Fetching transcript...")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        print(f"✅ Transcript loaded ({len(transcript)} characters)")
        
    except TranscriptsDisabled:
        print("❌ No captions available for this video.")
        return
    except Exception as e:
        error_msg = str(e)
        if "blocked" in error_msg.lower() or "ip" in error_msg.lower():
            print("❌ YouTube IP Blocking Detected")
            print("💡 Try using a different network or VPN")
        else:
            print(f"❌ Error fetching transcript: {error_msg}")
        return
    
    # Process transcript with LangChain
    print("🤖 Processing with AI...")
    try:
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
        
        print("✅ AI model ready!")
        
        # Interactive Q&A loop
        print("\n" + "=" * 40)
        print("💬 Ask questions about the video (type 'quit' to exit)")
        print("=" * 40)
        
        while True:
            question = input("\n❓ Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                print("Please enter a question.")
                continue
            
            try:
                print("🤔 Thinking...")
                answer = chain.invoke(question)
                print(f"\n💡 Answer: {answer}")
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                print("💡 Please try a different question.")
    
    except Exception as e:
        print(f"❌ Error setting up AI model: {str(e)}")
        print("💡 Please check your Google API key in .env file")

if __name__ == "__main__":
    main()