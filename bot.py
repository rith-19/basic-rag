import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import redis

# Load API key from environment variable
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
REDIS_URL = "YOUR_REDIS_URL"
SOURCE_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVER_K = 3

def setup_rag_chatbot():
    print("Loading embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print("Loading documents from web...")
    loader = WebBaseLoader(SOURCE_URL)
    docs = loader.load()
    
    print("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = splitter.split_documents(docs)
    print(f"Created document chunks")
    
    print("Creating vector store...")
    vs = InMemoryVectorStore.from_documents(documents=splits, embedding=embeddings)
    
    retriever = vs.as_retriever(search_kwargs={"k": RETRIEVER_K})
    
    llm = ChatGroq(model=LLM_MODEL, temperature=0.7)
    
    return retriever, llm

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def get_redis_history(session_id):
    """Get Redis chat history for a session"""
    return RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)

def create_rag_chain_with_memory(retriever, llm):
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided context.

IMPORTANT: Respond in natural, conversational language. Do NOT respond in JSON format or structured formats.

Context: {context}

Use the context above to answer the user's question in a friendly, natural way. If you cannot find the answer in the context, say so politely. If the user is just greeting you or having casual conversation, respond naturally without necessarily referencing the context."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    chain = (
        {
           "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
            "history": lambda x: x["history"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def chat_with_memory(chain, session_id="default"):
    history = get_redis_history(session_id)
    
    print("\nRAG Chatbot is ready! (Type 'quit' to exit)")
    print("Session ID:", session_id)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            # Prepare inputs with question and current history
            inputs = {
                "question": user_input,
                "history": history.messages  # Pass actual history messages
            }
            
            # Invoke the chain with proper inputs
            response = chain.invoke(inputs)
            
            # Add messages to history AFTER getting response
            history.add_user_message(user_input)
            history.add_ai_message(response)
            
            print(f"\nAssistant: {response}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    print("Setting up RAG chatbot...")
    
    # Check Redis connection
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print(" Connected to Redis successfully!")
    except redis.ConnectionError:
        print(" Failed to connect to Redis. Make sure Redis is running on localhost:6379")
        print("  You can start Redis with: docker run -d -p 6379:6379 redis")
        return
    
    # Setup RAG components
    retriever, llm = setup_rag_chatbot()
    
    # Create the chain using the dedicated function
    chain = create_rag_chain_with_memory(retriever, llm)
    
    print("\n Setup complete!")
    
    # Use the chat_with_memory function for the conversation loop
    session_id = "user_123"
    chat_with_memory(chain, session_id=session_id)

if __name__ == "__main__":
    main()
