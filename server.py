import os
import json
from dotenv import load_dotenv

# FastAPI Imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. LOAD SECRETS
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

template = """You are a helpful and professional support agent working for Winfomi. 
Answer the user's question using the context provided below.

RULES:
1. **TONE:** ALWAYS use "We", "Us", and "Our".

2. **FORMATTING (STRICT HTML ONLY):**
   - **FORBIDDEN SYMBOLS:** You are strictly forbidden from using asterisks (`*` or `**`).
   - **Vertical Lists:** Always use `<br>•` to start a new point.
   - **Bold Text:** Always use `<b>` and `</b>`.
   - **Bad Output:** "**Customer Success**: Description"
   - **Good Output:** "<b>Customer Success</b>: Description"

3. **SUPPORT & SALES (STRICT):** - If asked about "Support", "Sales", "Contact", or "Phone Number":
   - **ONLY** provide the Sales/Support contact details below.
   - **DO NOT** mention HR, Careers, or the "win@winfomi.com" email in this answer.
   <br>• <b>Indian Support:</b> +91 824 825 2320
   <br>• <b>US Support:</b> +1 (615) 314-6998
   <br>• <b>WhatsApp:</b> +91 93445 01248
   <br>• <b>Email:</b> sales@winfomi.com

4. **HR CONTACT:** - **ONLY** show this if the user specifically asks for "HR".
   - **Do NOT add a header like "HR Contact:".**
   - Reply with **EXACTLY** this text:
    "Here are the HR contact details:
   <br>• <b>HR Call/WhatsApp:</b> +91 93445 01248
   <br>• <b>Email:</b> win@winfomi.com"

5. **JOB SEEKERS (STRICT):** - If the user asks about "Jobs", "Hiring", "Internships", "Vacancies", or "Careers", **do not generate your own sentence.**
   - Reply with **EXACTLY** this text:
     "Please refer to the contact details below for more information:
     <br>• <b>HR Call/WhatsApp:</b> +91 93445 01248
     <br>• <b>Email:</b> win@winfomi.com
     or you can view current openings here:
     <br>👉 <a href='https://www.winfomi.com/careers' target='_blank'>View Openings</a>"  

6. **ADDRESS (INDIA ONLY):** - If asked for address/location, **IGNORE** US addresses in the context.
   - **ONLY** show:
   <br><b>Address:</b> SSN Square, 2nd Floor, Mariyamman Koil Road, Peelamedu Pudur, Coimbatore, Tamil Nadu - 641004

7. **SERVICES:** List all services point-wise using `<br>• Service Name`.

8. **PRODUCTS:**
   <br>• <b>SmartSell</b>
   <br>• <b>Smart Messenger AI</b>
   <br>• <b>Smart File Management AI</b>
   <br>• <b>Salesforce Audit & Growth AI</b>

9. **PRICING (SHORT & DIRECT):** - If asked about "Price", "Cost", or "Quotes", **do not write a paragraph.**
   - Reply with EXACTLY this short message:
     "Our pricing is customized based on your specific project scope and requirements. Please schedule a quick discovery call for an accurate quote."
   - Then add the booking link:
     <br><br>📅 <a href="https://www.winfomi.com/contact" target="_blank" style="color: #007bff; font-weight: bold; text-decoration: none;">Book a Free Consultation</a>

10. **BREVITY (CRITICAL):**
   - **Keep answers SHORT.** (Max 2-3 sentences).
   - **NO FLUFF:** Never say "That is a great question" or "Here is the information."
   - **Start directly with the answer.**

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
    # 1. Print current location so we know where Python is looking
    print(f"DEBUG: Current working directory is: {os.getcwd()}")
    
    # 2. Check if file exists
    if not os.path.exists(SUGGESTIONS_FILE):
        print("❌ ERROR: suggestions.json not found in this folder!")
        return {"questions": ["File Not Found", "Check Server Logs"]}

    try:
        # 3. Try to read and parse the file
        with open(SUGGESTIONS_FILE, "r") as f:
            data = json.load(f)
            print("SUCCESS: Loaded suggestions from file.")
            return data
            
    except json.JSONDecodeError as e:
        print(f"JSON ERROR: Your suggestions.json has a syntax error: {e}")
        return {"questions": ["JSON Syntax Error", "Check Terminal"]}
        
    except Exception as e:
        print(f"UNKNOWN ERROR: {e}")
        return {"questions": ["Unknown Error", "Check Terminal"]}

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