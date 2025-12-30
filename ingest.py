import os
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings 
from bs4 import BeautifulSoup as Soup
from dotenv import load_dotenv
import re

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "multilingual-e5-large"
NAMESPACE = "helper-agent"

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

# --- CONFIGURATION ---
START_URLS = ["https://www.winfomi.com/",
            "https://www.winfomi.com/services",
            "https://www.winfomi.com/products",
            "https://www.winfomi.com/about",
            "https://www.winfomi.com/contact"
            ]

# --- 1. CLEANING FUNCTION ---
def clean_html(content):
    soup = Soup(content, "html.parser")
    for tag in soup(["nav", "header", "footer", "script", "style", "aside", "form"]):
        tag.decompose()
    for div in soup.find_all("div", class_=re.compile(r"(menu|nav|sidebar|cookie|banner)")):
        div.decompose()
    return soup.get_text(separator=" ", strip=True)

# --- 2. INGEST FUNCTION ---
def ingest_data():
    # print("--- 1. CRAWLING {url}---")
    all_docs = []
    for url in START_URLS:
        print(f"--- CRAWLING: {url}---")
        try:
            loader = RecursiveUrlLoader(
                url=url,
                max_depth=1, 
                extractor=clean_html,
                prevent_outside=False, # Allow redirects (e.g., http -> https)
                timeout=10, 
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            docs = loader.load()
            valid_docs = [d for d in docs if "winfomi.com" in d.metadata['source']]
            
            print(f"     -> Found {len(valid_docs)} valid pages.")
            all_docs.extend(docs)
            print(f"Loaded {len(docs)} pages from {url}")
        except Exception as e:
            print(f"Failed to load {url}: {e}")

    # Remove duplicates
    unique_docs = {doc.metadata['source']: doc for doc in all_docs}.values()
    print(f"Total Unique Pages Scraped: {len(unique_docs)}")
    
    print("--- 2. CHUNKING ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(unique_docs)

    print("--- 3. UPLOADING TO PINECONE ---")

    embeddings = PineconeEmbeddings(
        model="multilingual-e5-large",
        pinecone_api_key=PINECONE_API_KEY
    )

    PineconeVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        namespace=NAMESPACE
    )
    print("Success! Data embedded and stored on Pinecone.")

if __name__ == "__main__":
    ingest_data()