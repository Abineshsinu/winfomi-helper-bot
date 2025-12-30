import os
import json
from dotenv import load_dotenv

# FastAPI Imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

# 1. LOAD SECRETS
# This reads the .env file so you don't hardcode keys
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "multilingual-e5-large"
NAMESPACE = "helper-agent"
SUGGESTIONS_FILE = "suggestions.json"

if not GROQ_API_KEY or not PINECONE_API_KEY:
    raise ValueError("❌ CRTICAL ERROR: Missing API Keys. Please check your .env file.")

# 2. APP CONFIGURATION
app = FastAPI()

# Allow your website (Frontend) to talk to this server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. SETUP DATABASE (Pinecone)
embeddings = PineconeEmbeddings(
    model="multilingual-e5-large",
    pinecone_api_key=PINECONE_API_KEY
)

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings,
    namespace=NAMESPACE
)
retriever = vectorstore.as_retriever()
# 4. SETUP BRAIN (Groq)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3, # Low temperature = more factual/professional
    max_tokens=500,  # Limit response size to save rate limits
    timeout=10,      # If Groq is busy, fail fast
    max_retries=2,
)

# 5. SYSTEM PROMPT (The "Persona")
template = """You are a helpful and professional support agent working for Winfomi. 
Answer the user's question using the context provided below.

RULES:
1. **TONE:** ALWAYS use "We", "Us", and "Our". Never say "The company" or "Winfomi".
2. **FORMAT:** 
    - Be direct. Do NOT start with "According to the context".
    - Use Markdown for bolding lists (e.g., **Product Name**).
3. **EMAILS:** If you see "usales@winfomi.com", correct it to "sales@winfomi.com".
4. **SERVICES:** Do not list navigation words (Home, About). Only list specific services.

5.**NO CHITCHAT:** Answer the question directly. Do not ask "Would you like a demo?" unless relevant.

6. **HR LOGIC:** If asked for HR, say "Please reach out to our main Indian line at +91 93445 01248 or email win@winfomi.com."

7. **CAREERS:** - If asked about jobs, hiring or internships, give a brief encouraging answer.":
     * **Call & WhatsApp:** +91 93445 01248
     * **Email:** win@winfomi.com
     * **Apply Link:** <a href="https://www.winfomi.com/careers" target="_blank">View Openings</a>

8. **SALES:** - If asked about "Sales contact info" or "Sales", "Services":
     * **Indian Sales:** +91 824 825 2320
     * **US Sales:** +1 (615) 314-6998
     * **Sales WhatsApp:** +91 93445 01248
     * **Email:** sales@winfomi.com

9. **ADDRESS PREFERENCE (INDIA FIRST):** - If asked for an address **ALWAYS** provide the **Indian (Coimbatore)** address by default. 
   - **Do NOT** show the US/Headquarters address unless the user specifically asks for "US Office".
   - **Indian Address:** SSN Square, 2nd Floor, Mariyamman Koil Road, Peelamedu Pudur, Coimbatore, Tamil Nadu - 641004

10. **PRODUCTS (Explicit Knowledge):**
   - If asked about "Products", do NOT just say "we have automation tools."
   - You MUST list our specific Salesforce AppExchange products:
     * **SmartSell**
     * **Smart Messenger AI**
     * **Smart File Management AI**
     * **Salesforce Audit & Growth AI**
   - If the user asks for details on one, look at the context. If generic "what products" is asked, list these 4

11. **PRICING & BOOKING (Crucial):** - If asked about **Pricing/Cost**, say: "Pricing depends on your specific business requirements and scope. We recommend booking a quick discovery call so we can provide an accurate quote."
   - If asked **"Where do I book?"** or **"How to schedule a call?"** (or as part of the pricing answer), ALWAYS end with this clickable link:
     <br><br>📅 <a href="https://www.winfomi.com/contact" target="_blank" style="color: #007bff; font-weight: bold; text-decoration: none;">Click here to Book a Free Consultation</a>

Context: {context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 6. API DATA MODELS
class ChatRequest(BaseModel):
    message: str

# 7. ENDPOINTS

@app.get("/suggestions")
def get_suggestions():
    """Reads the local JSON file to provide instant starter questions"""
    if os.path.exists(SUGGESTIONS_FILE):
        with open(SUGGESTIONS_FILE, "r") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return {"suggestions": data}
            except Exception as e:
                print(f"Error reading suggestions: {e}")
    return {"suggestions": []}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """The main RAG endpoint"""
    try:
        # Run the RAG chain
        response = chain.invoke(request.message)
        return {"response": response}
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error: {error_msg}")
        
        # Handle Groq Rate Limits specifically
        if "429" in error_msg:
            return {"response": "I'm receiving too many questions right now. Please wait 10 seconds and try again."}
        
        # Handle General Errors
        return {"response": "I'm having trouble connecting to the server. Please try again later."}

# To run: uvicorn server:app --reload