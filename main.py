from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Set
import uvicorn
import json
import os
import hashlib
import requests
from datetime import datetime
import sqlite3
from contextlib import contextmanager
import re
from urllib.parse import urljoin, urlparse, unquote, urlunparse
from bs4 import BeautifulSoup
import PyPDF2
import docx
import io
import logging
import shutil 
import time
import random
import logging as logger
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re


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

GOAL_KEYWORDS = {
    "Contact Us": ['contact', 'get in touch', 'address', 'phone', 'location', 'contact-us'],
    "Schedule a Demo": ['demo', 'schedule a call', 'book a meeting', 'request-a-demo'],
    "View Pricing": ['pricing', 'plans', 'subscribe', 'cost'],
    "Explore Products/Store": ['product', 'store', 'shop', 'buy', 'cart', 'checkout'],
    "Learn About Services": ['service', 'what we do', 'our work', 'solutions'],
    "Read the Blog/News": ['blog', 'news', 'article', 'update', 'stories'],
    "Access Help/Support": ['documentation', 'help', 'support', 'faq', 'knowledge base', 'knowledge-base'],
    "User Login/Account": ['login', 'signin', 'sign-in', 'my-account', 'my account'],
    "User Registration": ['register', 'signup', 'sign-up', 'create-account', 'create account'],
    "Explore Career Opportunities": ['career', 'job', 'hiring', 'work with us'],
    "Learn About the Company": ['about', 'about-us', 'our story', 'our team', 'company'],
}

def suggest_goals_from_content(sitemap_data: List[Dict]) -> List[str]:
    """
    Analyzes crawled content to suggest potential user goals.
    """
    found_goals = set()
    if not sitemap_data:
        return []

    # Combine all text, titles, and URLs for efficient searching
    full_text = " ".join(
        f"{p.get('url', '').lower()} {p.get('title', '').lower()} {p.get('content', ' ')}"
        for p in sitemap_data
    )

    for goal, keywords in GOAL_KEYWORDS.items():
        # Use regex for word boundaries to avoid partial matches (e.g., 'about' in 'about-us')
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', full_text) for keyword in keywords):
            found_goals.add(goal)

    # Limit the number of suggestions to a reasonable amount
    return sorted(list(found_goals))[:8]

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

def crawl_website(base_url: str, max_depth: int = 2, max_pages: int = 50) -> List[Dict]:
    """Crawl website and extract sitemap information with improved error handling"""
    
    # Basic validation and normalization of URL
    try:
        # Clean and validate base URL
        base_url = base_url.strip()
        if not base_url:
            logger.error("Empty base URL provided")
            return []
            
        parsed_url = urlparse(base_url)
        if not parsed_url.scheme:
            base_url = "https://" + base_url
            parsed_url = urlparse(base_url)
        
        if not parsed_url.netloc:
            logger.error(f"Invalid base URL: {base_url}")
            return []
            
        base_netloc = parsed_url.netloc.lower()
        
    except Exception as e:
        logger.error(f"Error parsing base URL {base_url}: {e}")
        return []

    visited: Set[str] = set()
    sitemap_data: List[Dict] = []
    session = requests.Session()
    
    # Rotate between different user agents to avoid blocking
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0'
    ]
    
    # Configure session with improved retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set session headers
    session.headers.update({
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Cache-Control': 'max-age=0'
    })

    def normalize_url(url: str) -> str:
        """Normalize URL for consistent comparison"""
        try:
            if not url:
                return ""
            
            url = url.strip()
            parsed = urlparse(url)
            
            # Remove fragment and normalize query parameters
            normalized = parsed._replace(fragment="")
            url_str = urlunparse(normalized)
            
            # Remove trailing slash except for root URLs  
            if url_str.endswith('/') and url_str.count('/') > 3:
                url_str = url_str[:-1]
                
            return url_str
        except Exception as e:
            logger.warning(f"Error normalizing URL {url}: {e}")
            return ""

    def is_valid_content_type(response) -> bool:
        """Check if response is HTML content"""
        content_type = response.headers.get('content-type', '').lower()
        return any(ct in content_type for ct in ['text/html', 'application/xhtml'])

    def extract_text_content(soup) -> str:
        """Extract and clean text content from soup"""
        try:
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
                element.decompose()
            
            # Extract text from relevant elements
            content_elements = soup.find_all([
                'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 
                'li', 'td', 'div', 'article', 'section', 'main'
            ])
            
            texts = []
            for element in content_elements:
                text = element.get_text(strip=True)
                if text and len(text) > 15:  # Filter out very short text
                    texts.append(text)
            
            content = " ".join(texts)
            # Clean up whitespace and normalize
            content = re.sub(r'\s+', ' ', content).strip()
            
            return content[:3000]  # Increased content length limit
            
        except Exception as e:
            logger.warning(f"Error extracting content: {e}")
            return ""

    def should_skip_url(url: str) -> bool:
        """Check if URL should be skipped based on common patterns"""
        url_lower = url.lower()
        skip_patterns = [
            '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico',
            '.zip', '.rar', '.tar', '.gz', '.doc', '.docx', '.xls', '.xlsx',
            '.ppt', '.pptx', '.mp3', '.mp4', '.avi', '.mov', '.wmv',
            '.css', '.js', '.xml', '.json', '.rss', '.atom',
            '/wp-admin/', '/admin/', '/login', '/register', '/dashboard/',
            '/api/', '/ajax/', '/search?', '/tag/', '/category/',
            '#', 'javascript:', 'mailto:', 'tel:', 'ftp:'
        ]
        
        return any(pattern in url_lower for pattern in skip_patterns)

    def crawl_page(url: str, depth: int = 0, parent_url: str = None):
        """Crawl individual page with comprehensive error handling"""
        
        # Check limits
        if len(sitemap_data) >= max_pages:
            logger.info(f"Reached maximum pages limit ({max_pages})")
            return
            
        if depth > max_depth:
            return

        normalized_url = normalize_url(url)
        if not normalized_url or normalized_url in visited:
            return

        # Skip URLs that are likely not content pages
        if should_skip_url(normalized_url):
            return

        visited.add(normalized_url)
        logger.info(f"Crawling: {normalized_url} (Depth: {depth})")

        try:
            # Add random delay to be respectful and avoid rate limiting
            if len(visited) > 1:
                time.sleep(random.uniform(0.5, 2.0))

            # Rotate user agent occasionally
            if len(visited) % 10 == 0:
                session.headers['User-Agent'] = random.choice(user_agents)

            # First, try a HEAD request to check if the resource exists
            try:
                head_response = session.head(
                    normalized_url,
                    timeout=(5, 10),
                    allow_redirects=True
                )
                
                # Check if it's likely HTML content
                content_type = head_response.headers.get('content-type', '').lower()
                if not any(ct in content_type for ct in ['text/html', 'application/xhtml', 'text/plain']):
                    logger.info(f"Skipping non-HTML content: {normalized_url} (Content-Type: {content_type})")
                    return
                    
            except Exception as head_error:
                logger.warning(f"HEAD request failed for {normalized_url}: {head_error}")
                # Continue with GET request anyway

            # Make the actual GET request
            response = session.get(
                normalized_url, 
                timeout=(10, 30),
                allow_redirects=True,
                stream=False
            )
            
            # Check response status
            if response.status_code == 403:
                logger.warning(f"Access forbidden (403) for: {normalized_url}")
                return
            elif response.status_code == 404:
                logger.warning(f"Page not found (404) for: {normalized_url}")
                return
            elif response.status_code >= 400:
                logger.warning(f"HTTP error {response.status_code} for: {normalized_url}")
                return
            
            response.raise_for_status()
            
            # Check content type again
            if not is_valid_content_type(response):
                logger.warning(f"Skipping non-HTML content: {normalized_url}")
                return
            
            # Check content size (avoid very large pages)
            content_length = len(response.content)
            if content_length > 10 * 1024 * 1024:  # 10MB limit
                logger.warning(f"Skipping large page ({content_length} bytes): {normalized_url}")
                return

            # Handle redirects - check if still within same domain
            final_url = normalize_url(response.url)
            if final_url != normalized_url:
                parsed_final = urlparse(response.url)
                if parsed_final.netloc.lower() != base_netloc:
                    logger.info(f"Redirect outside domain: {normalized_url} -> {response.url}")
                    return
                visited.add(final_url)
                normalized_url = final_url

            # Parse HTML with better error handling
            try:
                soup = BeautifulSoup(response.content, 'html.parser', from_encoding=response.encoding)
            except Exception as parse_error:
                logger.error(f"Error parsing HTML for {normalized_url}: {parse_error}")
                # Try with lxml parser as fallback
                try:
                    soup = BeautifulSoup(response.content, 'lxml', from_encoding=response.encoding)
                except:
                    # Final fallback to html.parser without encoding
                    soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title with better handling
            title_text = ""
            title_element = soup.find('title')
            if title_element:
                title_text = title_element.get_text().strip()
                title_text = re.sub(r'\s+', ' ', title_text)[:300]
            
            # Try meta description if no title
            if not title_text:
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    title_text = meta_desc.get('content').strip()[:300]

            # Extract content
            content = extract_text_content(soup)
            
            # Skip pages with very little content
            if len(content.strip()) < 50:
                logger.info(f"Skipping page with minimal content: {normalized_url}")
                return

            # Store page data
            page_data = {
                'url': normalized_url,
                'title': title_text,
                'content': content,
                'parent_url': parent_url,
                'depth': depth,
                'status_code': response.status_code,
                'content_length': len(content)
            }
            
            sitemap_data.append(page_data)
            logger.info(f"Successfully crawled: {normalized_url} (Title: {title_text[:50]}...)")

            # Find and crawl child links
            if depth < max_depth and len(sitemap_data) < max_pages:
                links_found = 0
                all_links = soup.find_all('a', href=True)
                
                # Shuffle links to get diverse content
                random.shuffle(all_links)
                
                for link in all_links:
                    if len(sitemap_data) >= max_pages:
                        break
                        
                    try:
                        href = link.get('href', '').strip()
                        if not href:
                            continue

                        # Skip obviously non-content links
                        if should_skip_url(href):
                            continue

                        # Resolve relative URLs
                        full_url = urljoin(normalized_url, href)
                        
                        # Clean URL (remove fragments and some query params)
                        clean_url = full_url.split('#')[0]
                        
                        parsed_link = urlparse(clean_url)
                        
                        # Only crawl internal links
                        if parsed_link.netloc.lower() == base_netloc:
                            crawl_page(clean_url, depth + 1, normalized_url)
                            links_found += 1
                            
                            # Limit links per page but be more generous
                            if links_found >= 30:
                                break
                                
                    except Exception as link_error:
                        logger.warning(f"Error processing link {href}: {link_error}")
                        continue

        except requests.exceptions.Timeout:
            logger.error(f"Timeout crawling {normalized_url}")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error crawling {normalized_url}: {e}")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error crawling {normalized_url}: {e}")
        except requests.exceptions.TooManyRedirects:
            logger.error(f"Too many redirects for {normalized_url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error crawling {normalized_url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error crawling {normalized_url}: {e}")

    # Start crawling with better initial URL handling
    try:
        # Try to crawl robots.txt first to respect crawling rules
        try:
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            robots_response = session.get(robots_url, timeout=20)
            if robots_response.status_code == 200:
                logger.info(f"Found robots.txt for {base_netloc}")
                # Basic robots.txt parsing could be added here
        except:
            pass  # Ignore robots.txt errors
            
        # Start crawling from the base URL
        crawl_page(base_url)
        
        # If we didn't get enough pages, try common paths
        if len(sitemap_data) < 5:
            common_paths = ['/about', '/contact', '/services', '/products', '/blog', '/news']
            for path in common_paths:
                if len(sitemap_data) >= max_pages:
                    break
                try_url = f"{parsed_url.scheme}://{parsed_url.netloc}{path}"
                crawl_page(try_url, 1, base_url)
                
    finally:
        session.close()
    
    logger.info(f"Crawling completed. Found {len(sitemap_data)} pages.")
    return sitemap_data


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
    """Generate sitemap by crawling the website and suggest user goals"""
    domain = request.domain
    logger.info(f"Starting sitemap crawl for domain: {domain}")
    
    try:
        sitemap_data = crawl_website(domain, max_depth=3, max_pages=100)
        logger.info(f"Finished crawling. Found {len(sitemap_data)} pages.")

        if not sitemap_data:
            logger.warning(f"No pages found for domain: {domain}")
            return {
                "message": "No pages found. The website might be blocking crawlers or have issues.", 
                "pages_found": 0, 
                "pages_saved_to_db": 0,
                "suggested_goals": []
            }

        # --- DYNAMIC GOAL SUGGESTION ---
        suggested_goals = suggest_goals_from_content(sitemap_data)
        logger.info(f"Suggested Goals based on content: {suggested_goals}")
        
        with get_db_connection() as conn:
            cursor = conn.cursor()

            # Delete existing sitemap data for this domain
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
                    continue
                    
            conn.commit()
            logger.info(f"Inserted {inserted_count} new sitemap entries into DB.")

        return {
            "message": "Sitemap generated and saved successfully", 
            "pages_found": len(sitemap_data), 
            "pages_saved_to_db": inserted_count,
            "suggested_goals": suggested_goals  # <-- Return the suggested goals
        }

    except Exception as e:
        logger.error(f"Error generating sitemap for {domain}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate sitemap: {str(e)}")

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
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            color: #e1e5f2;
            overflow: hidden;
        }
        
        /* Animated background particles */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: radial-gradient(circle at 20% 20%, rgba(120, 119, 198, 0.1) 0%, transparent 50%),
                        radial-gradient(circle at 80% 80%, rgba(255, 119, 198, 0.1) 0%, transparent 50%),
                        radial-gradient(circle at 40% 40%, rgba(120, 219, 255, 0.1) 0%, transparent 50%);
            animation: float 20s ease-in-out infinite;
            pointer-events: none;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            33% { transform: translateY(-10px) rotate(1deg); }
            66% { transform: translateY(5px) rotate(-1deg); }
        }
        
        .header {
            background: rgba(22, 22, 34, 0.9);
            backdrop-filter: blur(20px);
            padding: 16px 24px;
            border-bottom: 1px solid rgba(120, 119, 198, 0.2);
            display: flex;
            align-items: center;
            gap: 14px;
            position: relative;
            z-index: 10;
        }
        
        .header::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(120, 119, 198, 0.5), transparent);
        }
        
        .bot-avatar {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            color: white;
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
            position: relative;
            overflow: hidden;
        }
        
        .bot-avatar::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(45deg, transparent 30%, rgba(255, 255, 255, 0.1) 50%, transparent 70%);
            animation: shimmer 3s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
            100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
        }
        
        .header-info {
            flex: 1;
        }
        
        .header-info h3 {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 3px;
            background: linear-gradient(135deg, #e1e5f2, #a8b2d1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .status {
            font-size: 13px;
            color: #4ade80;
            display: flex;
            align-items: center;
            gap: 6px;
            font-weight: 500;
        }
        
        .status::before {
            content: '';
            width: 8px;
            height: 8px;
            background: #4ade80;
            border-radius: 50%;
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
            box-shadow: 0 0 8px rgba(74, 222, 128, 0.5);
        }
        
        @keyframes pulse {
            0%, 100% { 
                opacity: 1; 
                transform: scale(1);
            }
            50% { 
                opacity: 0.5; 
                transform: scale(1.1);
            }
        }
        
        .close-button {
            width: 32px;
            height: 32px;
            border: none;
            border-radius: 50%;
            background: rgba(239, 68, 68, 0.1);
            color: #ef4444;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(239, 68, 68, 0.2);
        }
        
        .close-button:hover {
            background: rgba(239, 68, 68, 0.2);
            transform: scale(1.1);
            box-shadow: 0 4px 16px rgba(239, 68, 68, 0.3);
        }
        
        .close-button:active {
            transform: scale(0.95);
        }
        
        .close-button svg {
            width: 18px;
            height: 18px;
        }
        
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            scroll-behavior: smooth;
            position: relative;
            z-index: 5;
        }
        
        .chat-container::-webkit-scrollbar {
            width: 8px;
        }
        
        .chat-container::-webkit-scrollbar-track {
            background: rgba(120, 119, 198, 0.1);
            border-radius: 4px;
        }
        
        .chat-container::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, rgba(120, 119, 198, 0.3), rgba(120, 119, 198, 0.6));
            border-radius: 4px;
            border: 1px solid rgba(120, 119, 198, 0.2);
        }
        
        .chat-container::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, rgba(120, 119, 198, 0.5), rgba(120, 119, 198, 0.8));
        }
        
        .message {
            max-width: 85%;
            padding: 16px 20px;
            border-radius: 20px;
            line-height: 1.5;
            font-size: 15px;
            position: relative;
            animation: messageSlide 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            word-wrap: break-word;
            overflow-wrap: break-word;
            backdrop-filter: blur(10px);
        }
        
        @keyframes messageSlide {
            from {
                opacity: 0;
                transform: translateY(20px) scale(0.95);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }
        
        .user-message {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 8px;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        .user-message::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        }
        
        .bot-message {
            background: rgba(22, 22, 34, 0.8);
            backdrop-filter: blur(20px);
            color: #e1e5f2;
            align-self: flex-start;
            border-bottom-left-radius: 8px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(120, 119, 198, 0.2);
            position: relative;
            overflow: hidden;
        }
        
        .bot-message::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(120, 119, 198, 0.3), transparent);
        }
        
        .typing-indicator {
            display: none;
            align-self: flex-start;
            background: rgba(22, 22, 34, 0.8);
            backdrop-filter: blur(20px);
            padding: 16px 20px;
            border-radius: 20px;
            border-bottom-left-radius: 8px;
            max-width: 100px;
            border: 1px solid rgba(120, 119, 198, 0.2);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        
        .typing-dots {
            display: flex;
            gap: 6px;
            justify-content: center;
        }
        
        .typing-dots span {
            width: 8px;
            height: 8px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 50%;
            animation: typing 1.6s infinite ease-in-out;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
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
                transform: translateY(-12px);
                opacity: 1;
            }
        }
        
        .input-container {
            background: rgba(22, 22, 34, 0.9);
            backdrop-filter: blur(20px);
            padding: 20px 24px;
            border-top: 1px solid rgba(120, 119, 198, 0.2);
            display: flex;
            gap: 16px;
            align-items: flex-end;
            position: relative;
            z-index: 10;
        }
        
        .input-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(120, 119, 198, 0.5), transparent);
        }
        
        .input-wrapper {
            flex: 1;
            position: relative;
        }
        
        .message-input {
            width: 100%;
            min-height: 48px;
            max-height: 120px;
            padding: 14px 20px;
            border: 2px solid rgba(120, 119, 198, 0.2);
            border-radius: 24px;
            font-size: 15px;
            font-family: inherit;
            resize: none;
            outline: none;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            background: rgba(16, 16, 30, 0.8);
            backdrop-filter: blur(10px);
            color: #e1e5f2;
            box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.3);
        }
        
        .message-input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.2), inset 0 2px 8px rgba(0, 0, 0, 0.3);
            background: rgba(16, 16, 30, 0.9);
        }
        
        .message-input::placeholder {
            color: #6b7280;
        }
        
        .send-button {
            width: 48px;
            height: 48px;
            border: none;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
            position: relative;
            overflow: hidden;
        }
        
        .send-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), transparent);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .send-button:hover:not(:disabled) {
            transform: translateY(-2px) scale(1.05);
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.5);
        }
        
        .send-button:hover:not(:disabled)::before {
            opacity: 1;
        }
        
        .send-button:active:not(:disabled) {
            transform: translateY(0) scale(0.98);
        }
        
        .send-button:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none;
        }
        
        .send-button svg {
            width: 22px;
            height: 22px;
            z-index: 1;
        }
        
        .welcome-message {
            text-align: center;
            padding: 24px;
            color: rgba(225, 229, 242, 0.8);
            font-size: 18px;
            font-weight: 500;
            background: rgba(22, 22, 34, 0.6);
            border-radius: 16px;
            border: 1px solid rgba(120, 119, 198, 0.2);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        }
        
        .error-message {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
            align-self: flex-start;
            border-bottom-left-radius: 8px;
            box-shadow: 0 8px 32px rgba(239, 68, 68, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .retry-button {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
            padding: 8px 16px;
            border-radius: 16px;
            font-size: 13px;
            cursor: pointer;
            margin-top: 10px;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .retry-button:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-1px);
        }
        
        @media (max-width: 400px) {
            .message {
                max-width: 90%;
                font-size: 14px;
                padding: 14px 18px;
            }
            
            .header {
                padding: 14px 18px;
            }
            
            .input-container {
                padding: 16px 18px;
            }
            
            .chat-container {
                padding: 18px;
            }
            
            .bot-avatar {
                width: 36px;
                height: 36px;
                font-size: 18px;
            }
            
            .header-info h3 {
                font-size: 16px;
            }
        }
        
        /* Smooth scrollbar for Firefox */
        .chat-container {
            scrollbar-width: thin;
            scrollbar-color: rgba(120, 119, 198, 0.5) rgba(120, 119, 198, 0.1);
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
        <button class="close-button" id="closeButton" onclick="closeWidget()">
            <svg viewBox="0 0 24 24" fill="currentColor">
                <path d="M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12L19,6.41Z" />
            </svg>
        </button>
    </div>
    
    <div class="chat-container" id="chatContainer">
        <div class="welcome-message">
            Hello! I'm your navigation helper. How can I assist you today?
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
        
        function closeWidget() {
            // Check if this is running in an iframe (widget mode)
            if (window.parent !== window) {
                // Send message to parent window to close the widget
                window.parent.postMessage({ action: 'closeWidget' }, '*');
            } else {
                // If not in iframe, just close the window/tab
                window.close();
            }
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