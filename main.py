#!/usr.bin/env python3
"""
Navigation Helper Bot - Backend API with RAG
Complete backend system for the navigation helper bot with RAG capabilities
using Langchain and ChromaDB.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import json
import os
import hashlib
import requests
from datetime import datetime
import sqlite3
from contextlib import contextmanager
import re
from urllib.parse import urljoin, urlparse, unquote
from bs4 import BeautifulSoup
import PyPDF2
import docx
import io
import logging
import shutil # Added for Chroma DB directory cleanup

# --- Langchain / RAG Imports ---
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings # Or use HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI # Or use another LLM via langchain_community
from langchain_core.documents import Document # For creating Langchain Document objects

# --- Environment Variables ---
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file

# Ensure OpenAI API key is set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable not set!")
    # In a real application, you might want to raise an exception or exit
    # For this example, we'll proceed but RAG features will fail without it.
    # raise EnvironmentError("OPENAI_API_KEY environment variable not set.")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Navigation Helper Bot API (RAG Enabled)", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Setup (for metadata and logs) ---
DATABASE_FILE = "nav_bot.db"

def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    # Knowledge base table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            file_hash TEXT UNIQUE
        )
    ''')

    # Sitemap table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sitemap (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            title TEXT,
            content TEXT,
            parent_url TEXT,
            depth INTEGER DEFAULT 0,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Bot configurations table
    # Using SQL comments (--) instead of Python comments (#) within the SQL string
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bot_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain TEXT UNIQUE NOT NULL, -- Ensure unique domain
            widget_position TEXT DEFAULT 'bottom-right',
            widget_color TEXT DEFAULT '#007bff',
            goals TEXT, -- JSON string of goals
            dns_verified BOOLEAN DEFAULT FALSE,
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Chat logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Customer information table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS customer_info (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            name TEXT,
            email TEXT,
            phone TEXT,
            additional_info TEXT,
            collected_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row # Allows accessing columns by name
    try:
        yield conn
    finally:
        conn.close()

# --- RAG System Setup ---
CHROMA_DB_PATH = "./chroma_db"
rag_chain = None # Global variable to hold the RAG chain

def initialize_rag_system():
    """Initializes the Langchain RAG components."""
    if not OPENAI_API_KEY:
        logger.warning("Skipping RAG initialization: OPENAI_API_KEY not set.")
        return

    try:
        logger.info("Starting RAG system initialization...")

        # Initialize Embedding Model
        logger.info("Initializing OpenAI Embeddings...")
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        logger.info("OpenAI Embeddings initialized.")


        # Initialize Chroma Vector Store
        # If directory exists, load it. Otherwise, an empty one will be created on first add.
        logger.info(f"Loading or creating Chroma DB at {CHROMA_DB_PATH}...")
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        logger.info("Chroma DB initialized.")


        # Initialize LLM
        logger.info("Initializing ChatOpenAI LLM...")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=OPENAI_API_KEY) # Using a cost-effective model
        logger.info("LLM initialized.")

        # Create a Retriever
        # Configure the retriever: how many relevant documents to fetch (k)
        logger.info("Creating Retriever...")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        logger.info("Retriever created.")

        # Create the RAG chain
        logger.info("Creating RetrievalQA chain...")
        global rag_chain
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False # Set to True to see source chunks in response
        )
        logger.info("RetrievalQA chain initialized.")

        logger.info("RAG system initialization complete.")


    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        rag_chain = None # Ensure rag_chain is None if initialization fails

def add_document_to_rag(file_content: bytes, filename: str, file_type: str):
    """Processes file content, splits it, and adds to the RAG vector store."""
    if not rag_chain:
        logger.warning("RAG system not initialized. Cannot add document.")
        return False

    try:
        # Save content to a temporary file to use Langchain loaders
        # Langchain loaders typically work with file paths
        temp_dir = "./temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_filepath = os.path.join(temp_dir, filename)
        with open(temp_filepath, "wb") as f:
            f.write(file_content)

        documents = []
        # Use appropriate Langchain loader
        if file_type == 'pdf':
            loader = PyPDFLoader(temp_filepath)
        elif file_type == 'docx':
            loader = Docx2txtLoader(temp_filepath)
        elif file_type == 'txt':
             # Use standard TextLoader, specifying encoding
            loader = TextLoader(temp_filepath, encoding="utf-8")
        else:
            logger.warning(f"Unsupported file type for RAG processing: {file_type}")
            os.remove(temp_filepath) # Clean up temp file
            return False

        logger.info(f"Loading documents from {filename}...")
        loaded_docs = loader.load()
        logger.info(f"Loaded {len(loaded_docs)} pages/sections.")

        # Split text into chunks
        logger.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(loaded_docs)
        logger.info(f"Split into {len(split_docs)} chunks.")

        if not split_docs:
            logger.warning("No text chunks generated from the document.")
            os.remove(temp_filepath)
            return False

        # Add chunks to vector store
        # We need access to the underlying vectorstore, not just the chain
        # Re-initialize vectorstore here just to get the instance connected to the path
        # A better way in a larger app is to pass the vectorstore instance around or make it part of the RAGSystem class
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

        logger.info(f"Adding {len(split_docs)} chunks to Chroma DB...")
        # Add metadata to chunks if desired (e.g., filename, source page)
        # Langchain loaders often add source/page metadata automatically
        # vectorstore.add_documents([Document(page_content=chunk.page_content, metadata={'source': filename}) for chunk in split_docs])
        vectorstore.add_documents(split_docs) # Loaders usually add relevant metadata
        logger.info("Chunks added successfully.")

        os.remove(temp_filepath) # Clean up temp file
        # Clean up temp directory if empty
        if not os.listdir(temp_dir):
             os.rmdir(temp_dir)

        return True

    except Exception as e:
        logger.error(f"Error processing document {filename} for RAG: {e}")
        # Attempt to clean up temp file if it exists
        if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
             os.remove(temp_filepath)
        return False

# --- Helper functions (kept from original, slightly modified or used differently) ---

def generate_file_hash(content: bytes) -> str:
    """Generate SHA-256 hash of file content"""
    return hashlib.sha256(content).hexdigest()

def crawl_website(base_url: str, max_depth: int = 2) -> List[Dict]: # Reduced max_depth for faster example
    """Crawl website and extract sitemap information"""
    # Basic validation and normalization of URL
    try:
        parsed_url = urlparse(base_url)
        if not parsed_url.scheme:
            base_url = "https://" + base_url # Assume https if no scheme
            parsed_url = urlparse(base_url)
        if not parsed_url.netloc:
             logger.error(f"Invalid base URL: {base_url}")
             return []
        base_netloc = parsed_url.netloc
    except Exception as e:
        logger.error(f"Error parsing base URL {base_url}: {e}")
        return []


    visited = set()
    sitemap_data = []
    session = requests.Session() # Use a session for potential connection pooling

    def crawl_page(url: str, depth: int = 0, parent_url: str = None):
        # Normalize URL for consistent comparison (remove trailing slashes, fragments, queries)
        normalized_url = urlparse(url)._replace(fragment="", query="").geturl()
        if normalized_url.endswith('/'):
             normalized_url = normalized_url[:-1]

        if depth > max_depth or normalized_url in visited:
            return

        visited.add(normalized_url)
        logger.info(f"Crawling: {normalized_url} (Depth: {depth})")

        try:
            # Add headers to mimic a browser and avoid potential blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; NavHelperBot/1.0; +http://your-website.com/bot)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5'
            }
            response = session.get(normalized_url, timeout=15, headers=headers, allow_redirects=True) # Increased timeout
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            # Handle redirects and update visited set if necessary
            if response.url != normalized_url and urlparse(response.url).netloc == base_netloc:
                 visited.add(urlparse(response.url)._replace(fragment="", query="").geturl().rstrip('/'))


            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""

            # Extract main content (basic extraction from common tags)
            content_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td'])
            content = " ".join(ele.get_text() for ele in content_elements).strip()
            # Basic cleanup: replace multiple spaces/newlines with single space
            content = re.sub(r'\s+', ' ', content)


            sitemap_data.append({
                'url': normalized_url,
                'title': title_text,
                'content': content[:1000],  # Limit content length stored in DB
                'parent_url': parent_url,
                'depth': depth
            })

            # Find all links for further crawling
            if depth < max_depth:
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    # Handle relative URLs, protocol-relative URLs, and fragments/queries
                    full_url = urljoin(normalized_url, href).split('#')[0].split('?')[0] # Remove fragments/queries before parsing
                    parsed_link_url = urlparse(full_url)

                    # Only crawl internal links within the base domain
                    if parsed_link_url.netloc == base_netloc:
                        crawl_page(full_url, depth + 1, normalized_url)

        except requests.exceptions.RequestException as req_e:
            logger.error(f"Request error crawling {normalized_url}: {req_e}")
        except Exception as e:
            logger.error(f"Unexpected error crawling {normalized_url}: {e}")


    crawl_page(base_url)
    return sitemap_data


# --- Pydantic models (kept as is) ---
class ChatMessage(BaseModel):
    message: str
    session_id: str

class CustomerInfo(BaseModel):
    session_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    additional_info: Optional[str] = None

class BotConfig(BaseModel):
    domain: str
    widget_position: str = "bottom-right"
    widget_color: str = "#007bff"
    goals: List[str] = [] # Stored as JSON string in DB

class DNSVerification(BaseModel):
    domain: str

class WidgetRequest(BaseModel):
    domain: str

class SitemapRequest(BaseModel):
    domain: str


# --- API Routes ---

@app.on_event("startup")
async def startup_event():
    """Initialize database and RAG system on startup"""
    init_database()
    initialize_rag_system()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources if necessary (e.g., temp files)"""
    temp_dir = "./temp_uploads"
    if os.path.exists(temp_dir):
        logger.info(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)
    # Note: ChromaDB handles persistence itself, no explicit shutdown needed usually

@app.get("/")
async def root():
    return {"message": "Navigation Helper Bot API with RAG is running"}

@app.post("/api/upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents to knowledge base (process for RAG)"""
    uploaded_files_info = []
    successful_rag_additions = 0

    if not rag_chain:
         raise HTTPException(status_code=500, detail="RAG system is not initialized. Cannot process documents.")

    with get_db_connection() as conn:
        cursor = conn.cursor()

        for file in files:
            filename = file.filename
            logger.info(f"Processing file: {filename}")
            try:
                content = await file.read()
                file_hash = generate_file_hash(content)

                # Check if file already exists by hash
                cursor.execute("SELECT id FROM knowledge_base WHERE file_hash = ?", (file_hash,))
                if cursor.fetchone():
                    logger.info(f"File {filename} with hash {file_hash} already exists. Skipping.")
                    uploaded_files_info.append({
                        'filename': filename,
                        'status': 'skipped (duplicate)',
                        'size': len(content)
                    })
                    continue # Skip duplicate files

                # Determine file type and add to RAG
                if filename.lower().endswith('.pdf'):
                    file_type = 'pdf'
                elif filename.lower().endswith('.docx'):
                    file_type = 'docx'
                elif filename.lower().endswith('.txt'):
                    file_type = 'txt'
                else:
                    logger.warning(f"Unsupported file type for upload: {filename}")
                    uploaded_files_info.append({
                        'filename': filename,
                        'status': 'skipped (unsupported type)',
                        'size': len(content)
                    })
                    continue # Skip unsupported file types

                # Add document content to the RAG vector store
                rag_added = add_document_to_rag(content, filename, file_type)

                if rag_added:
                    # Insert file metadata into database
                    cursor.execute("""
                        INSERT INTO knowledge_base (filename, file_type, file_hash)
                        VALUES (?, ?, ?)
                    """, (filename, file_type, file_hash))
                    conn.commit() # Commit immediately for this file

                    uploaded_files_info.append({
                        'filename': filename,
                        'type': file_type,
                        'size': len(content),
                        'status': 'uploaded and processed'
                    })
                    successful_rag_additions += 1
                else:
                    uploaded_files_info.append({
                        'filename': filename,
                        'type': file_type,
                        'size': len(content),
                        'status': 'failed RAG processing'
                    })


            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")
                uploaded_files_info.append({
                    'filename': filename,
                    'status': f'failed ({e})',
                    'size': len(content) if 'content' in locals() else 0
                })
                # Rollback the transaction for this file if any error occurred before commit
                conn.rollback() # This rolls back the cursor's operations since the last commit


    return {
        "message": "Document upload processing complete",
        "results": uploaded_files_info,
        "successful_rag_additions": successful_rag_additions
    }

@app.post("/api/generate-sitemap")
async def generate_sitemap(request: SitemapRequest):
    """Generate sitemap by crawling the website"""
    domain = request.domain
    logger.info(f"Starting sitemap crawl for domain: {domain}")
    try:
        sitemap_data = crawl_website(domain)
        logger.info(f"Finished crawling. Found {len(sitemap_data)} pages.")

        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Delete existing sitemap data for this domain (or perhaps all?)
            # Decided to clear all for simplicity in this example, adjust if needed
            # For production, might want to associate sitemaps with domains
            cursor.execute("DELETE FROM sitemap")
            logger.info("Cleared existing sitemap data from DB.")

            # Insert new sitemap data
            inserted_count = 0
            for page in sitemap_data:
                try:
                    cursor.execute("""
                        INSERT INTO sitemap (url, title, content, parent_url, depth)
                        VALUES (?, ?, ?, ?, ?)
                    """, (page['url'], page['title'], page['content'],
                         page['parent_url'], page['depth']))
                    inserted_count += 1
                except sqlite3.Error as db_error:
                    logger.error(f"Database error inserting sitemap page {page['url']}: {db_error}")
                    conn.rollback() # Rollback insertion of this page if error occurs
            conn.commit()
            logger.info(f"Inserted {inserted_count} new sitemap entries into DB.")

        return {"message": "Sitemap generated and saved successfully", "pages_found": len(sitemap_data), "pages_saved_to_db": inserted_count}

    except Exception as e:
        logger.error(f"Error generating sitemap for {domain}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate sitemap: {e}")

@app.post("/api/chat")
async def chat(message: ChatMessage):
    print("................", message)
    """Handle chat messages using RAG"""
    logger.info(f"Received chat message from session {message.session_id}: {message.message}")
    user_message = message.message

    # Fallback/Initial Welcome Response
    if not rag_chain:
         logger.warning("RAG system not available. Providing fallback response.")
         bot_response = "Sorry, the document knowledge base is not currently available. How else can I help you navigate?"
         # You might add other simple rule-based responses here if needed
    else:
        try:
            # Use the RAG chain to get the response
            logger.info(f"Querying RAG system with: {user_message}")
            # The .invoke() method is common for Langchain Runnables (Chains are Runnables)
            response_object = rag_chain.invoke({"query": user_message}) # Input key might vary based on chain type
            bot_response = response_object.get("result", "Could not find a relevant answer in the documents.") # Extract the actual response text

            logger.info(f"RAG system response: {bot_response[:100]}...") # Log snippet

        except Exception as e:
            logger.error(f"Error during RAG query for session {message.session_id}: {e}")
            bot_response = "Sorry, I encountered an error while trying to find an answer. Please try again or ask differently."

    # Log the conversation regardless of RAG success
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_logs (session_id, user_message, bot_response)
                VALUES (?, ?, ?)
            """, (message.session_id, user_message, bot_response))
            conn.commit()
    except sqlite3.Error as db_error:
        logger.error(f"Error logging chat message to DB: {db_error}")


    return {"response": bot_response}

@app.post("/api/collect-customer-info")
async def collect_customer_info(info: CustomerInfo):
    """Collect customer information"""
    logger.info(f"Collecting customer info for session {info.session_id}")
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Check if info for this session_id already exists
            cursor.execute("SELECT id FROM customer_info WHERE session_id = ?", (info.session_id,))
            existing_info = cursor.fetchone()

            if existing_info:
                # Update existing record
                 cursor.execute("""
                     UPDATE customer_info
                     SET name = COALESCE(?, name),
                         email = COALESCE(?, email),
                         phone = COALESCE(?, phone),
                         additional_info = COALESCE(?, additional_info),
                         collected_date = CURRENT_TIMESTAMP
                     WHERE session_id = ?
                 """, (info.name, info.email, info.phone, info.additional_info, info.session_id))
                 message = "Customer information updated successfully"
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO customer_info (session_id, name, email, phone, additional_info)
                    VALUES (?, ?, ?, ?, ?)
                """, (info.session_id, info.name, info.email, info.phone, info.additional_info))
                message = "Customer information collected successfully"

            conn.commit()

        return {"message": message}

    except sqlite3.Error as e:
        logger.error(f"Error collecting customer info: {e}")
        raise HTTPException(status_code=500, detail=f"Database error collecting customer info: {e}")
    except Exception as e:
        logger.error(f"Unexpected error collecting customer info: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/api/configure-bot")
async def configure_bot(config: BotConfig):
    """Configure bot settings"""
    logger.info(f"Configuring bot for domain: {config.domain}")
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Use INSERT OR REPLACE based on domain uniqueness
            cursor.execute("""
                INSERT OR REPLACE INTO bot_config (domain, widget_position, widget_color, goals)
                VALUES (?, ?, ?, ?)
            """, (config.domain, config.widget_position, config.widget_color,
                 json.dumps(config.goals))) # Store goals list as JSON string
            conn.commit()

        return {"message": f"Bot configuration saved successfully for {config.domain}"}

    except sqlite3.Error as e:
        logger.error(f"Error configuring bot: {e}")
        raise HTTPException(status_code=500, detail=f"Database error configuring bot: {e}")
    except Exception as e:
        logger.error(f"Unexpected error configuring bot: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/api/verify-dns")
async def verify_dns(verification: DNSVerification):
    """Verify DNS record for domain (basic check)"""
    domain = verification.domain.replace('http://', '').replace('https://', '').split('/')[0] # Clean up domain
    logger.info(f"Attempting to verify domain: {domain}")
    # Basic verification - try to access the domain over HTTPS
    # A proper DNS verification might involve checking for a specific TXT record
    try:
        # Use a HEAD request which is lighter than GET
        response = requests.head(f"https://{domain}", timeout=15, verify=True) # Add verify=True for SSL checks

        # Consider success codes (2xx) and potentially redirects (3xx)
        if 200 <= response.status_code < 400:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                # Ensure a config exists for this domain before updating
                cursor.execute("SELECT id FROM bot_config WHERE domain = ?", (domain,))
                if cursor.fetchone():
                    cursor.execute("""
                        UPDATE bot_config SET dns_verified = TRUE WHERE domain = ?
                    """, (domain,))
                    conn.commit()
                    logger.info(f"Domain {domain} verified successfully.")
                    return {"verified": True, "message": "Domain verified successfully"}
                else:
                     logger.warning(f"Domain {domain} verified but no config found in DB.")
                     return {"verified": True, "message": "Domain verified, but no configuration found for this domain."} # Verified but no config
        else:
            logger.warning(f"Domain {domain} verification failed with status code: {response.status_code}")
            return {"verified": False, "message": f"Domain verification failed: Received status code {response.status_code}"}

    except requests.exceptions.SSLError as ssl_e:
         logger.error(f"SSL error verifying DNS for {domain}: {ssl_e}")
         return {"verified": False, "message": f"SSL/TLS error verifying domain: {ssl_e}. Ensure HTTPS is properly configured."}
    except requests.exceptions.RequestException as req_e:
        logger.error(f"Request error verifying DNS for {domain}: {req_e}")
        return {"verified": False, "message": f"Could not reach domain: {req_e}. Ensure domain is correct and accessible."}
    except Exception as e:
        logger.error(f"Unexpected error verifying DNS for {domain}: {e}")
        return {"verified": False, "message": f"Verification error: {str(e)}"}

@app.post("/api/widget-code")
async def get_widget_code(request: WidgetRequest):
    """Generate widget code for integration"""
    domain = request.domain
    
    # Get bot configuration if exists
    widget_position = "bottom-right"
    widget_color = "#007bff"
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT widget_position, widget_color FROM bot_config WHERE domain = ?", (domain,))
            config = cursor.fetchone()
            if config:
                widget_position = config['widget_position']
                widget_color = config['widget_color']
    except Exception as e:
        logger.error(f"Error fetching bot config: {e}")
    
    # Position mapping for CSS
    position_styles = {
        "bottom-right": "bottom: 20px; right: 20px;",
        "bottom-left": "bottom: 20px; left: 20px;",
        "top-right": "top: 20px; right: 20px;",
        "top-left": "top: 20px; left: 20px;"
    }
    
    position_css = position_styles.get(widget_position, position_styles["bottom-right"])
    
    widget_code = f"""
    <!-- Navigation Helper Bot Widget -->
    <div id="nav-helper-bot"></div>
    <script>
        (function() {{
            var botConfig = {{
                domain: '{domain}',
                apiUrl: 'http://localhost:8000',
                position: '{widget_position}',
                color: '{widget_color}'
            }};
            
            // Create widget container
            var botContainer = document.createElement('div');
            botContainer.id = 'nav-bot-container';
            botContainer.style.cssText = `
                position: fixed;
                {position_css}
                z-index: 9999;
                width: 380px;
                height: 600px;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                border-radius: 16px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.2);
                display: none;
                overflow: hidden;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                border: 1px solid rgba(255, 255, 255, 0.3);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            `;
            
            // Create toggle button
            var botToggle = document.createElement('button');
            botToggle.innerHTML = '💬';
            botToggle.style.cssText = `
                position: fixed;
                {position_css}
                z-index: 10000;
                width: 64px;
                height: 64px;
                border-radius: 50%;
                border: none;
                background: linear-gradient(135deg, {widget_color}, #0056b3);
                color: white;
                font-size: 28px;
                cursor: pointer;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                display: flex;
                align-items: center;
                justify-content: center;
            `;
            
            // Add hover effects
            botToggle.addEventListener('mouseenter', function() {{
                this.style.transform = 'translateY(-2px)';
                this.style.boxShadow = '0 6px 24px rgba(0,0,0,0.2)';
            }});
            
            botToggle.addEventListener('mouseleave', function() {{
                this.style.transform = 'translateY(0)';
                this.style.boxShadow = '0 4px 20px rgba(0,0,0,0.15)';
            }});
            
            // Toggle functionality
            var isOpen = false;
            botToggle.onclick = function() {{
                if (!isOpen) {{
                    botContainer.style.display = 'block';
                    setTimeout(() => {{
                        botContainer.style.opacity = '1';
                        botContainer.style.transform = 'scale(1)';
                    }}, 10);
                    botToggle.innerHTML = '✕';
                    isOpen = true;
                }} else {{
                    botContainer.style.opacity = '0';
                    botContainer.style.transform = 'scale(0.95)';
                    setTimeout(() => {{
                        botContainer.style.display = 'none';
                    }}, 300);
                    botToggle.innerHTML = '💬';
                    isOpen = false;
                }}
            }};
            
            // Create iframe
            var iframe = document.createElement('iframe');
            iframe.src = botConfig.apiUrl + '/widget?domain=' + encodeURIComponent(botConfig.domain);
            iframe.style.cssText = 'width: 100%; height: 100%; border: none; border-radius: 16px;';
            iframe.setAttribute('title', 'Navigation Helper Bot');
            
            // Add close on outside click
            document.addEventListener('click', function(e) {{
                if (isOpen && !botContainer.contains(e.target) && !botToggle.contains(e.target)) {{
                    botToggle.click();
                }}
            }});
            
            // Add escape key handler
            document.addEventListener('keydown', function(e) {{
                if (e.key === 'Escape' && isOpen) {{
                    botToggle.click();
                }}
            }});
            
            // Responsive adjustments
            function adjustForMobile() {{
                if (window.innerWidth < 480) {{
                    botContainer.style.width = '95vw';
                    botContainer.style.height = '85vh';
                    botContainer.style.left = '2.5vw';
                    botContainer.style.right = 'auto';
                    botContainer.style.bottom = '2vh';
                    botContainer.style.top = 'auto';
                }}
            }}
            
            window.addEventListener('resize', adjustForMobile);
            adjustForMobile();
            
            botContainer.appendChild(iframe);
            document.body.appendChild(botContainer);
            document.body.appendChild(botToggle);
        }})();
    </script>
    """
    
    return {"widget_code": widget_code, "domain": domain}

@app.get("/api/widget-code/{domain:path}")
async def get_widget_code_get(domain: str):
    """Generate widget code for integration (GET method with URL decoding)"""
    decoded_domain = unquote(domain)
    
    # Get bot configuration
    widget_position = "bottom-right"
    widget_color = "#007bff"
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT widget_position, widget_color FROM bot_config WHERE domain = ?", (decoded_domain,))
            config = cursor.fetchone()
            if config:
                widget_position = config['widget_position']
                widget_color = config['widget_color']
    except Exception as e:
        logger.error(f"Error fetching bot config: {e}")
    
    # Reuse the POST endpoint logic
    request = WidgetRequest(domain=decoded_domain)
    return await get_widget_code(request)

@app.get("/widget")
async def widget_interface(domain: str = None):
    """Return the improved widget HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Navigation Helper Bot</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                height: 100vh;
                display: flex;
                flex-direction: column;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }
            
            .header {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                padding: 15px 20px;
                border-bottom: 1px solid rgba(0, 0, 0, 0.1);
                display: flex;
                align-items: center;
                gap: 12px;
            }
            
            .bot-avatar {
                width: 36px;
                height: 36px;
                background: linear-gradient(135deg, #667eea, #764ba2);
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 18px;
                color: white;
            }
            
            .header-info h3 {
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 2px;
            }
            
            .status {
                font-size: 12px;
                color: #28a745;
                display: flex;
                align-items: center;
                gap: 4px;
            }
            
            .status::before {
                content: '';
                width: 8px;
                height: 8px;
                background: #28a745;
                border-radius: 50%;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }
            
            .chat-container {
                flex: 1;
                overflow-y: auto;
                padding: 20px;
                display: flex;
                flex-direction: column;
                gap: 16px;
                scroll-behavior: smooth;
            }
            
            .chat-container::-webkit-scrollbar {
                width: 6px;
            }
            
            .chat-container::-webkit-scrollbar-track {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
            }
            
            .chat-container::-webkit-scrollbar-thumb {
                background: rgba(255, 255, 255, 0.3);
                border-radius: 3px;
            }
            
            .message {
                max-width: 85%;
                padding: 12px 16px;
                border-radius: 18px;
                line-height: 1.4;
                font-size: 14px;
                position: relative;
                animation: messageSlide 0.3s ease-out;
                word-wrap: break-word;
                overflow-wrap: break-word;
            }
            
            @keyframes messageSlide {
                from {
                    opacity: 0;
                    transform: translateY(10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .user-message {
                background: linear-gradient(135deg, #007bff, #0056b3);
                color: white;
                align-self: flex-end;
                border-bottom-right-radius: 6px;
                box-shadow: 0 2px 8px rgba(0, 123, 255, 0.3);
            }
            
            .bot-message {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                color: #333;
                align-self: flex-start;
                border-bottom-left-radius: 6px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .typing-indicator {
                display: none;
                align-self: flex-start;
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                padding: 12px 16px;
                border-radius: 18px;
                border-bottom-left-radius: 6px;
                max-width: 85px;
            }
            
            .typing-dots {
                display: flex;
                gap: 4px;
            }
            
            .typing-dots span {
                width: 8px;
                height: 8px;
                background: #666;
                border-radius: 50%;
                animation: typing 1.4s infinite ease-in-out;
            }
            
            .typing-dots span:nth-child(2) {
                animation-delay: 0.2s;
            }
            
            .typing-dots span:nth-child(3) {
                animation-delay: 0.4s;
            }
            
            @keyframes typing {
                0%, 60%, 100% {
                    transform: translateY(0);
                    opacity: 0.4;
                }
                30% {
                    transform: translateY(-10px);
                    opacity: 1;
                }
            }
            
            .input-container {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                padding: 16px 20px;
                border-top: 1px solid rgba(0, 0, 0, 0.1);
                display: flex;
                gap: 12px;
                align-items: flex-end;
            }
            
            .input-wrapper {
                flex: 1;
                position: relative;
            }
            
            .message-input {
                width: 100%;
                min-height: 44px;
                max-height: 120px;
                padding: 12px 16px;
                border: 2px solid rgba(0, 0, 0, 0.1);
                border-radius: 22px;
                font-size: 14px;
                font-family: inherit;
                resize: none;
                outline: none;
                transition: all 0.2s ease;
                background: white;
            }
            
            .message-input:focus {
                border-color: #007bff;
                box-shadow: 0 0 0 3px rgba(0, 123, 255, 0.1);
            }
            
            .message-input::placeholder {
                color: #999;
            }
            
            .send-button {
                width: 44px;
                height: 44px;
                border: none;
                border-radius: 50%;
                background: linear-gradient(135deg, #007bff, #0056b3);
                color: white;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s ease;
                box-shadow: 0 2px 8px rgba(0, 123, 255, 0.3);
            }
            
            .send-button:hover:not(:disabled) {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(0, 123, 255, 0.4);
            }
            
            .send-button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            .send-button svg {
                width: 20px;
                height: 20px;
            }
            
            .welcome-message {
                text-align: center;
                padding: 20px;
                color: rgba(255, 255, 255, 0.9);
                font-size: 16px;
                font-weight: 500;
            }
            
            .error-message {
                background: linear-gradient(135deg, #dc3545, #c82333);
                color: white;
                align-self: flex-start;
                border-bottom-left-radius: 6px;
                box-shadow: 0 2px 8px rgba(220, 53, 69, 0.3);
            }
            
            .retry-button {
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
                color: white;
                padding: 6px 12px;
                border-radius: 12px;
                font-size: 12px;
                cursor: pointer;
                margin-top: 8px;
                transition: all 0.2s ease;
            }
            
            .retry-button:hover {
                background: rgba(255, 255, 255, 0.3);
            }
            
            @media (max-width: 400px) {
                .message {
                    max-width: 90%;
                    font-size: 13px;
                }
                
                .header {
                    padding: 12px 16px;
                }
                
                .input-container {
                    padding: 12px 16px;
                }
                
                .chat-container {
                    padding: 16px;
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <div class="bot-avatar">🤖</div>
            <div class="header-info">
                <h3>Navigation Helper</h3>
                <div class="status">Online</div>
            </div>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="welcome-message">
                👋 Hello! I'm your navigation helper. How can I assist you today?
            </div>
        </div>
        
        <div class="typing-indicator" id="typingIndicator">
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
        
        <div class="input-container">
            <div class="input-wrapper">
                <textarea 
                    id="messageInput" 
                    class="message-input" 
                    placeholder="Type your message..." 
                    rows="1"
                ></textarea>
            </div>
            <button class="send-button" id="sendButton" onclick="sendMessage()">
                <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M2,21L23,12L2,3V10L17,12L2,14V21Z" />
                </svg>
            </button>
        </div>
        
        <script>
            const urlParams = new URLSearchParams(window.location.search);
            const domain = urlParams.get('domain');
            let sessionId = Math.random().toString(36).substring(7);
            let retryCount = 0;
            const MAX_RETRIES = 3;
            
            const chatContainer = document.getElementById('chatContainer');
            const messageInput = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            const typingIndicator = document.getElementById('typingIndicator');
            
            // Auto-resize textarea
            messageInput.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = Math.min(this.scrollHeight, 120) + 'px';
                sendButton.disabled = !this.value.trim();
            });
            
            // Send message on Enter
            messageInput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
            
            function addMessage(content, isUser = false, isError = false, showRetry = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user-message' : (isError ? 'error-message' : 'bot-message')}`;
                
                // Handle HTML content safely
                if (typeof content === 'string') {
                    messageDiv.textContent = content;
                } else {
                    messageDiv.innerHTML = content;
                }
                
                if (showRetry && isError) {
                    const retryBtn = document.createElement('button');
                    retryBtn.className = 'retry-button';
                    retryBtn.textContent = 'Retry';
                    retryBtn.onclick = () => {
                        const lastUserMessage = Array.from(chatContainer.children)
                            .filter(el => el.classList.contains('user-message'))
                            .pop();
                        if (lastUserMessage) {
                            sendMessage(lastUserMessage.textContent);
                        }
                    };
                    messageDiv.appendChild(retryBtn);
                }
                
                chatContainer.appendChild(messageDiv);
                scrollToBottom();
                return messageDiv;
            }
            
            function showTypingIndicator() {
                chatContainer.appendChild(typingIndicator);
                typingIndicator.style.display = 'block';
                scrollToBottom();
            }
            
            function hideTypingIndicator() {
                typingIndicator.style.display = 'none';
                if (typingIndicator.parentNode) {
                    typingIndicator.parentNode.removeChild(typingIndicator);
                }
            }
            
            function scrollToBottom() {
                setTimeout(() => {
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }, 100);
            }
            
            async function sendMessage(messageText = null) {
                const message = messageText || messageInput.value.trim();
                if (!message || sendButton.disabled) return;
                
                sendButton.disabled = true;
                messageInput.disabled = true;
                
                if (!messageText) {
                    addMessage(message, true);
                    messageInput.value = '';
                    messageInput.style.height = 'auto';
                }
                
                showTypingIndicator();
                
                try {
                    const requestBody = {
                        message: message,
                        session_id: sessionId
                    };
                    
                    if (domain) {
                        requestBody.domain = domain;
                    }
                    
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(requestBody)
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    
                    hideTypingIndicator();
                    addMessage(data.response || 'I apologize, but I encountered an issue processing your request.');
                    retryCount = 0; // Reset retry count on success
                    
                } catch (error) {
                    console.error('Error:', error);
                    hideTypingIndicator();
                    
                    let errorMessage = 'Sorry, there was an error processing your message.';
                    let showRetry = false;
                    
                    if (retryCount < MAX_RETRIES) {
                        errorMessage += ' Please try again.';
                        showRetry = true;
                        retryCount++;
                    } else {
                        errorMessage += ' Please refresh the page and try again.';
                    }
                    
                    addMessage(errorMessage, false, true, showRetry);
                } finally {
                    messageInput.disabled = false;
                    messageInput.focus();
                    sendButton.disabled = !messageInput.value.trim();
                }
            }
            
            // Initialize
            document.addEventListener('DOMContentLoaded', function() {
                messageInput.focus();
                sendButton.disabled = true;
                
                // Send initial greeting if domain is provided
                if (domain) {
                    console.log('Widget loaded for domain:', domain);
                }
            });
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.get("/api/knowledge-base")
async def get_knowledge_base():
    """Get all documents metadata in knowledge base"""
    logger.info("Received GET request for /api/knowledge-base") # Added log
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Select metadata only, content is in Chroma
            cursor.execute("SELECT filename, file_type, upload_date FROM knowledge_base ORDER BY upload_date DESC")
            documents = [dict(row) for row in cursor.fetchall()]
        logger.info(f"Returning {len(documents)} documents metadata.") # Added log
        return {"documents": documents}
    except Exception as e:
         logger.error(f"Error fetching knowledge base metadata: {e}") # Added error log
         raise HTTPException(status_code=500, detail=f"Error fetching knowledge base metadata: {e}")

@app.get("/api/sitemap")
async def get_sitemap():
    """Get generated sitemap (from SQLite)"""
    logger.info("Received GET request for /api/sitemap") # Added log
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT url, title, depth FROM sitemap ORDER BY depth, url")
            sitemap = [dict(row) for row in cursor.fetchall()]
        logger.info(f"Returning {len(sitemap)} sitemap entries.") # Added log
        return {"sitemap": sitemap}
    except Exception as e:
         logger.error(f"Error fetching sitemap: {e}") # Added error log
         raise HTTPException(status_code=500, detail=f"Error fetching sitemap: {e}")

@app.delete("/api/knowledge-base")
async def clear_knowledge_base():
    """Deletes all documents metadata from SQLite and clears Chroma DB."""
    logger.info("Received DELETE request for /api/knowledge-base") # Added log
    if not rag_chain:
         raise HTTPException(status_code=500, detail="RAG system is not initialized. Cannot clear knowledge base.")

    try:
        # Clear SQLite metadata
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM knowledge_base")
            conn.commit()
            logger.info("Cleared knowledge_base table in SQLite.")

        # Clear Chroma DB directory
        if os.path.exists(CHROMA_DB_PATH):
            logger.info(f"Clearing Chroma DB directory: {CHROMA_DB_PATH}")
            shutil.rmtree(CHROMA_DB_PATH)
            # Re-initialize RAG chain to connect to the new empty DB
            initialize_rag_system() # Re-initialize RAG system after clearing Chroma
            logger.info("Chroma DB cleared and RAG system re-initialized.")
        else:
             logger.warning(f"Chroma DB directory not found at {CHROMA_DB_PATH}. Nothing to clear.")

        return {"message": "Knowledge base cleared successfully."}

    except sqlite3.Error as db_error:
        logger.error(f"Database error clearing knowledge base: {db_error}")
        raise HTTPException(status_code=500, detail=f"Database error clearing knowledge base: {db_error}")
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.get("/api/status")
async def get_status():
    """Provides status of the RAG system and database."""
    logger.info("Received GET request for /api/status") # Added log
    rag_status = "Initialized" if rag_chain else "Not Initialized (Check logs for errors like missing API key)"

    db_status = "Connected"
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM knowledge_base")
            kb_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM sitemap")
            sitemap_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM bot_config")
            config_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM chat_logs")
            chat_log_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM customer_info")
            customer_info_count = cursor.fetchone()[0]
        logger.info("Database counts retrieved successfully.") # Added log
    except Exception as e:
        db_status = f"Error: {e}"
        kb_count = sitemap_count = config_count = chat_log_count = customer_info_count = "Error" # Change counts to Error on failure
        logger.error(f"Error fetching database counts: {e}") # Added error log


    # Attempt to get Chroma count (might fail if not initialized or empty)
    chroma_count = 0
    try:
        if rag_chain:
             # Access the underlying vectorstore from the chain
             # This is a bit hacky; ideally, the vectorstore would be a direct global or passed around
             vectorstore = rag_chain.retriever.vectorstore
             # Note: ChromaDB might raise an error if the collection doesn't exist yet or is empty
             try:
                 chroma_count = vectorstore._collection.count()
             except Exception as chroma_e:
                 logger.warning(f"Could not get Chroma count (collection error?): {chroma_e}")
                 chroma_count = "Error/Empty?" # Indicate potential issue getting count
        else:
             chroma_count = "RAG not initialized"
    except Exception as e:
         logger.warning(f"Could not get Chroma count (general error): {e}")
         chroma_count = "Error/Not Available"


    return {
        "status": "OK",
        "rag_system": rag_status,
        "database": db_status,
        "counts": {
            "knowledge_base_metadata": kb_count, # Count in SQLite
            "knowledge_base_chunks_in_chroma": chroma_count, # Count in Chroma
            "sitemap_entries": sitemap_count,
            "bot_configurations": config_count,
            "chat_logs": chat_log_count,
            "customer_info": customer_info_count
        }
    }

if __name__ == "__main__":
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 