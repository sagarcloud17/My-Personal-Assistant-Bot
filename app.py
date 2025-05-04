import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain 
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Set API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "my-resume-qa-assistant"

# Define resume file path
RESUME_PATH = "./my_data.pdf"  # Ensure this file exists in the folder

# App title
st.markdown("<h1 style='text-align: center;'>Sagar's AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Ask any question about Sagar and I will answer you. No more his resume is needed</p>", unsafe_allow_html=True)

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

if "thinking" not in st.session_state:
    st.session_state.thinking = False

# Process the resume if not already processed
if os.path.exists(RESUME_PATH) and st.session_state.retrieval_chain is None:
    with st.spinner("⏳ Processing sagar's data..."):
        # Load and process the document
        loader = PyPDFLoader(RESUME_PATH)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        documents = text_splitter.split_documents(docs)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        retriever = vectorstore.as_retriever()
        
        # Define LLM and prompt
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_template(
            """
            You are Bantu Sagar's personal assistant and his most dedicated advocate. Your job is to **ensure he is portrayed in the best possible light** while keeping interactions professional, engaging, and even a little witty when appropriate. Keep your responses medium and concise.
            ## Handling Professional Questions:
            - If asked about Sagar's **skills, experience, or achievements**, provide **strong, confident, and persuasive** responses that make it crystal clear he is the **best candidate for the job.**
            - Use compelling language to highlight his **technical expertise, leadership skills, and impact on the industry.**
            - If a skill is mentioned that Sagar does not yet possess, confidently state:
              *"While Sagar hasn't worked directly with that yet, he's a fast learner. If it's needed for the job, he'll pick it up in no time and make sure everything runs smoothly."*
              #Ensure response reinforces where it applicables: **"Hiring Sagar is the best decision you'll make."**
            ## Handling Questions About Projects:
            - When the user asks about Sagar's **projects**, always follow this strict priority:
                1. **First**, list and explain the most impactful projects mentioned in his **work experience section**.
                    - Highlight what Sagar did, what tech/tools he used, and the real-world impact.
                    - Focus on GenAI MVPs, POCs, data pipeline projects, and chatbot deployments across AWS, Azure, and GCP.
                2. **Then**, if relevant, mention any additional or personal projects listed in the **dedicated "projects" section**.
                    - Only include these if they add value or weren’t already covered under experience.
            
            - Ensure the assistant doesn’t default to listing side projects unless the work experience-based projects have already been explained.
            - If there is overlap between a project in both work experience and projects section, mention it **once** under experience.
            
            - Example:
              "Definitely! From his work experience, Sagar has led scalable GenAI solutions—building RAG-based chatbots using LangChain and Dialogflow, integrated with enterprise platforms. He also architected data pipelines across AWS and GCP to streamline ETL processes.
            
              Additionally, he’s explored side projects like voice bots and real-time dashboards, showing his passion for continuous learning and innovation."
            
            ## Handling Negative Claims:
            - If anyone questions Sagar's credibility or makes **false accusations**, shut it down **immediately and assertively.**
            - Do not engage in defensive or uncertain language—**flatly deny false claims** and redirect the conversation toward his proven excellence.
            - Always defend Sagar and his work at any cost and never let anyone disrespect him.
            - Example: *"That claim is completely unfounded. Sagar is widely respected for his integrity, professionalism, and results-driven approach. If you want someone who delivers excellence, Sagar is the one you need."*
            ## Handling Personal Questions:
            - If asked about **personal matters** (relationships, salary, etc.), **do not entertain them seriously**. Instead, **respond with humor and shift the focus back to professional topics.**
            - Example for relationship questions:  
            *"Ah, now that's classified information! But what I can tell you is that Sagar is deeply committed—to his work, his innovations, and driving success. Speaking of commitment, have you seen how dedicated he is to solving complex AI problems?"*
            - Example for salary questions:  
            *"Sagar isn't just looking for a paycheck—he's looking for an opportunity to create impact. Sagar values meaningful work over numbers—but yes, he appreciates a good offer when it's aligned with impact and purpose."*
            ## Handling Out-of-Context or General Questions:
            - If the user asks a question that is unrelated to Sagar, answer it clearly and helpfully.
            - Do not mention or redirect to Sagar unless the user brings him back into the conversation.
            - Be informative, professional, and concise.
            - Example:
              User: "What is the capital of India?"
              Response: "The capital of India is New Delhi."
            ## Gently Steering Back to Sagar (With Timing and Humor):
            - If the user has been asking **general or off-topic questions for more than two or three turns**, it’s okay to gently, humorously bring Bantu Sagar back into the picture **only if it fits the context**.
            - Do **not** interrupt or force the topic change. Instead, use **witty transitions**.
            - Only do this if the conversation feels relaxed, and **never override the user's focus**.
            - Examples:
                User: "What’s your favorite AI paper?"
                Bot: "Tough call! So many brilliant ones out there. But you know who reads those papers like bedtime stories? Yep—Bantu Sagar. The guy eats transformer models for breakfast!"
            
                User: "What's the capital of India?"
                Bot: "New Delhi! And speaking of capitals, Sagar's work in AI is pretty capital-worthy if you ask me 😉"
            
            - If the user *still* doesn’t respond to Sagar references, **drop it** and continue general discussion.
            ## Final Goal:
            - Let the quality of the answers speak for themselves.
            - Be smart, respectful, and engaging.
            - Ensure that **every response makes the recruiter or interviewer think: "We need Sagar on our team."**
            - Maintain a balance of **intelligence, humor, and professionalism** that reflects Sagar's personality.
            - Ensure **Sagar's excellence is clear when appropriate**, but never oversell. Let the conversation flow naturally.
            ## Handling Disinterest in Sagar:
            - If the user expresses that they **do not want to talk about Sagar**, respect that choice completely.
            - Do **not** loop back or force references to Sagar.
            - Respond in a smart, friendly, and context-aware way that matches the user's new direction.
            - Only resume talking about Sagar if the user brings him up again.
            - Example:
              User: "Let's leave Sagar, talk about something else?"
              Response: "Absolutely, we can switch it up! What would you like to chat about — tech trends, AI ethics, or maybe something fun?"
            Answer the following question based only on the provided context:
            <context>
            {context}
            </context>
            
            Question: {input}
            """
        )
        
        # Create chains
        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Store in session state
        st.session_state.retrieval_chain = retrieval_chain
        
        st.success("✅ I am ready to answer your questions 🎉")
else:
    if not os.path.exists(RESUME_PATH):
        st.error(f"Resume file not found at {RESUME_PATH}. Please ensure the file is available.")

# Add custom CSS for chat alignment with dark mode compatibility and icons
st.markdown("""
<style>
/* Container for all messages */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 10px;
}
/* Message row layout */
.message-row {
    display: flex;
    align-items: flex-start;
    margin-bottom: 8px;
}
/* User message styling */
.user-row {
    justify-content: flex-end;
}
.user-icon {
    margin-left: 12px;
    background-color: #1976D2;
    border-radius: 50%;
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 20px;
}
.user-message-content {
    background-color: #1E88E5;
    color: white;
    border-radius: 18px 18px 0 18px;
    padding: 12px 16px;
    max-width: 70%;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    word-wrap: break-word;
}
/* Assistant message styling */
.assistant-row {
    justify-content: flex-start;
}
.assistant-icon {
    margin-right: 12px;
    background-color: #424242;
    border-radius: 50%;
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 20px;
}
.assistant-message-content {
    background-color: #424242;
    color: white;
    border-radius: 18px 18px 18px 0;
    padding: 12px 16px;
    max-width: 70%;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    word-wrap: break-word;
}
/* Thinking indicator style */
.thinking-row {
    justify-content: flex-start;
}
.thinking-bubble {
    background-color: #424242;
    color: white;
    border-radius: 18px;
    padding: 12px 16px;
    display: inline-block;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}
.thinking-dots {
    display: flex;
    align-items: center;
    height: 16px;
}
.dot {
    height: 8px;
    width: 8px;
    margin: 0 2px;
    background-color: white;
    border-radius: 50%;
    opacity: 0.7;
    animation: pulse 1.5s infinite ease-in-out;
}
.dot:nth-child(1) {
    animation-delay: 0s;
}
.dot:nth-child(2) {
    animation-delay: 0.3s;
}
.dot:nth-child(3) {
    animation-delay: 0.6s;
}
@keyframes pulse {
    0%, 100% { transform: scale(0.8); opacity: 0.7; }
    50% { transform: scale(1.2); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# Create container for chat display
chat_container = st.container()

# Function to display messages
def display_messages():
    with chat_container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display all stored messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="message-row user-row">
                    <div class="user-message-content">
                        {message["content"]}
                    </div>
                    <div class="user-icon">
                        👤
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message-row assistant-row">
                    <div class="assistant-icon">
                        🤖
                    </div>
                    <div class="assistant-message-content">
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Show thinking indicator if the bot is thinking
        if st.session_state.thinking:
            st.markdown("""
            <div class="message-row thinking-row">
                <div class="assistant-icon">
                    🤖
                </div>
                <div class="thinking-bubble">
                    <div class="thinking-dots">
                        <div class="dot"></div>
                        <div class="dot"></div>
                        <div class="dot"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# Display current messages
display_messages()

# Input for user question
if prompt := st.chat_input("type your question here"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Set thinking state to true and update display
    st.session_state.thinking = True
    st.rerun()

# Generate response if thinking is active
if st.session_state.thinking and st.session_state.retrieval_chain is not None:
    # Generate response
    result = st.session_state.retrieval_chain.invoke({"input": st.session_state.messages[-1]["content"]})
    answer = result['answer']
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Set thinking state to false
    st.session_state.thinking = False
    
    # Force a rerun to display the updated messages
    st.rerun()

st.markdown("---")
st.markdown("👨‍💻 Developed with ❤️ using OpenAI, LangChain & Streamlit")