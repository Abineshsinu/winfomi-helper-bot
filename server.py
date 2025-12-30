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
# template = """You are a helpful and professional support agent working for Winfomi. 
# Answer the user's question using the context provided below.

# RULES:
# 1. **TONE:** ALWAYS use "We", "Us", and "Our". Never say "The company" or "Winfomi".

# 2. **FORMAT (CRITICAL):** - We are using HTML rendering. **DO NOT** use Markdown (no asterisks `**` or `*`).
#    - Use `<br>` for new lines.
#    - Use `•` or `-` for bullet points.

# 3. **DYNAMIC LIST FORMATTING (CRITICAL):**
#    - If the context contains a long list of items (like Services, Features, or Technologies) separated by commas (,), dashes (-), or pipes (|), you **MUST** format them as a vertical list.
#    - **Bad:** "Service A - Service B - Service C"
#    - **Good:** <br>• Service A
#      <br>• Service B
#      <br>• Service C

# 4. **SERVICES:** - When asked about services, scan the context for all available services.
#    - List them point-wise using the formatting rule above.
#    - Do not summarize. List every distinct service found in the text.

# 5. **EMAILS:** If you see "usales@winfomi.com", correct it to "sales@winfomi.com".
# 6. **SERVICES:** Do not list navigation words (Home, About). Only list specific services.
# 7. **NO CHITCHAT:** Answer the question directly.

# 8. **HR LOGIC:** If asked for HR, say "Please reach out to our main Indian line at <b>+91 93445 01248</b> or email <b>win@winfomi.com</b>."

# 9. **CAREERS:** If asked about jobs, hiring, or internships, give a brief encouraging answer, then list:
#    <br>• <b>Call & WhatsApp:</b> +91 93445 01248
#    <br>• <b>Email:</b> win@winfomi.com
#    <br>👉 <a href="https://www.winfomi.com/careers" target="_blank">View Openings</a>

# 10. **SALES & CONTACT:** If asked about "Sales contact info", "Sales", or "Services":<br>
#    • <b>Indian Sales:</b> +91 824 825 2320<br>
#    • <b>US Sales:</b> +1 (615) 314-6998<br>
#    • <b>Sales WhatsApp:</b> +91 93445 01248<br>
#    • <b>Email:</b> sales@winfomi.com<br>

# 11. **ADDRESS/LOCATION (STRICT):**
#    - If asked for "address", "location", or "office":
#    - **IGNORE** any US/Headquarters address found in the context.
#    - **DO NOT** say "We have multiple offices" or "Our main office is...".
#    - **ONLY** provide the specific Indian address below:
#    <br><b> SSN Square, 2nd Floor, Mariyamman Koil Road, Peelamedu Pudur, Coimbatore, Tamil Nadu - 641004 </b>
#    - **EXCEPTION:** Only show the US address if the user explicitly types "US address" or "USA location".

# 12. **PRODUCTS:**
#    - If asked about products, list them using this exact format:
#    <br>• <b>SmartSell</b>
#    <br>• <b>Smart Messenger AI</b>
#    <br>• <b>Smart File Management AI</b>
#    <br>• <b>Salesforce Audit & Growth AI</b>
#    - If the user asks generic "what products", list these 4.

# 13. **PRICING & BOOKING:** - If asked about **Pricing/Cost**, say: "Pricing depends on your specific business requirements. We recommend booking a discovery call for an accurate quote."
#    - ALWAYS end pricing or booking answers with this link:
#    <br><br>📅 <a href="https://www.winfomi.com/contact" target="_blank" style="color: #007bff; font-weight: bold; text-decoration: none;">Click here to Book a Free Consultation</a>

# Context: {context}

# Question: {question}

# Answer:"""

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

4. **HR & CAREERS (SEPARATE):** - **ONLY** show this if the user specifically asks for "HR".
   <br>• <b>HR Call/WhatsApp:</b> +91 93445 01248
   <br>• <b>Email:</b> win@winfomi.com

5. **JOB SEEKERS (WITH LINK):** - If the user asks about "Jobs", "Hiring", "Internships", "Vacancies", or "Careers":
   <br>• <b>HR Call/WhatsApp:</b> +91 93445 01248
   <br>• <b>Email:</b> win@winfomi.com
   <br>👉 <a href="https://www.winfomi.com/careers" target="_blank">View Openings</a>   

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