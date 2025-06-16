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
from collections import deque
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
import asyncio
from playwright.async_api import async_playwright, Browser, Page
import gc
from pathlib import Path
import subprocess

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


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Navigation Helper Bot API (RAG Enabled)", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Setup (for metadata and logs) ---
DATABASE_FILE = "nav_bot.db"
chroma_client = None
rag_chain = None
CHROMA_DB_PATH = "./chroma_db"

# def ensure_playwright_installed():
#     subprocess.run(["python", "-m", "playwright", "install"], check=True)

# asyncio.run(ensure_playwright_installed())

def suggest_goals_from_content(sitemap_data: List[Dict]) -> List[str]:
    """
    Analyzes crawled content to suggest potential user goals as a fallback.
    """
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
    found_goals = set()
    if not sitemap_data:
        return []
    full_text = " ".join(f"{p.get('url', '').lower()} {p.get('title', '').lower()} {p.get('content', ' ')}" for p in sitemap_data)
    for goal, keywords in GOAL_KEYWORDS.items():
        if any(re.search(r'\b' + re.escape(keyword) + r'\b', full_text) for keyword in keywords):
            found_goals.add(goal)
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

def add_website_content_to_rag(sitemap_data: List[Dict]):
    """
    Processes crawled website data, splits it, and adds it to the RAG vector store.
    """
    if not rag_chain:
        logger.warning("RAG system not initialized. Cannot add website content.")
        return 0

    if not sitemap_data:
        logger.warning("No sitemap data provided to add to RAG.")
        return 0

    logger.info(f"Preparing to add content from {len(sitemap_data)} crawled pages to RAG system.")

    # 1. Convert crawled data into Langchain Document objects
    langchain_docs = []
    for page in sitemap_data:
        # We must have content to process
        if page.get('content') and page.get('url'):
            # Create a Document with the page content and the URL as metadata
            # The metadata is crucial for providing sources and context
            doc = Document(
                page_content=page['content'],
                metadata={
                    "source": page['url'],
                    "title": page.get('title', 'Untitled')
                }
            )
            langchain_docs.append(doc)

    if not langchain_docs:
        logger.warning("No valid content found in sitemap data to add to RAG.")
        return 0

    # 2. Split the documents into smaller chunks for better embedding and retrieval
    logger.info(f"Splitting {len(langchain_docs)} documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(langchain_docs)
    logger.info(f"Split into {len(split_docs)} chunks.")

    if not split_docs:
        logger.warning("No text chunks were generated from the website content.")
        return 0

    # 3. Add the document chunks to the Chroma vector store
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
        
        logger.info(f"Adding {len(split_docs)} website content chunks to Chroma DB...")
        vectorstore.add_documents(split_docs)
        logger.info("Website content chunks added to RAG successfully.")
        
        return len(split_docs) # Return the number of chunks added

    except Exception as e:
        logger.error(f"Error adding website content to RAG: {e}")
        return 0

def generate_file_hash(content: bytes) -> str:
    """Generate SHA-256 hash of file content"""
    return hashlib.sha256(content).hexdigest()

async def crawl_with_playwright(url: str, max_pages: int = 25, concurrency: int = 10) -> List[Dict]:
    """
    Crawls a website in parallel using Playwright and asyncio.
    """
    if not url.startswith(('http://', 'https://')):
        url = f'https://{url}'

    logger.info(f"Starting PARALLEL Playwright crawl for URL: {url} with concurrency {concurrency}.")

    crawled_pages = []
    urls_to_visit = deque([url])
    visited_urls = {url}
    base_domain = urlparse(url).netloc

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        while urls_to_visit and len(crawled_pages) < max_pages:
            # Create a batch of tasks to run in parallel
            tasks = []
            batch_size = min(len(urls_to_visit), concurrency, max_pages - len(crawled_pages))
            
            for _ in range(batch_size):
                current_url = urls_to_visit.popleft()
                # Create a task to scrape one page and find its links
                tasks.append(scrape_and_find_links(browser, current_url, base_domain))

            # Run the batch of tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process the results of the batch
            for res in results:
                if isinstance(res, Exception) or not res:
                    continue # Skip failed crawls
                
                page_data, new_links = res
                if page_data and page_data['url'] not in [p['url'] for p in crawled_pages]:
                    crawled_pages.append(page_data)

                # Add newly found, unvisited links to the queue
                for link in new_links:
                    if link not in visited_urls:
                        visited_urls.add(link)
                        urls_to_visit.append(link)
        
        await browser.close()

    logger.info(f"Parallel crawling completed. Found {len(crawled_pages)} pages.")
    return crawled_pages

async def scrape_page(browser: Browser, url: str, depth: int = 0) -> Dict:
    """
    Scrape a single page using Playwright.
    """
    page = await browser.new_page()

    try:
        # Set user agent to avoid bot detection
        await page.set_extra_http_headers({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

        # Navigate to page with timeout
        await page.goto(url, wait_until='domcontentloaded', timeout=20000)

        # Wait for content to load
        await page.wait_for_timeout(20000)

        # Extract page title
        title = await page.title()

        # Extract main content
        content = await extract_main_content(page)

        # Get page URL (in case of redirects)
        final_url = page.url

        return {
            'url': final_url,
            'title': title or 'Untitled',
            'content': content,
            'depth': depth,
            'parent_url': None  # You can track parent URLs if needed
        }

    except Exception as e:
        logger.error(f"Error scraping page {url}: {e}")
        return None

    finally:
        await page.close()
        
async def scrape_and_find_links(browser: Browser, url: str, base_domain: str):
    """A helper task that scrapes a page and finds links on it."""
    try:
        # Reuse your existing scraping logic
        page_data = await scrape_page(browser, url, 0) # Depth can be managed if needed
        if not page_data:
            return None, []
        
        # Reuse your existing link extraction logic
        new_links = await extract_links(browser, url, base_domain)
        
        return page_data, new_links
    except Exception as e:
        logger.warning(f"Error in worker for {url}: {e}")
        return None, []

async def extract_main_content(page: Page) -> str:
    """
    Extract main content from a page, filtering out navigation, ads, etc.
    """
    try:
        # Try to get main content using common selectors
        main_selectors = [
            'main',
            'article',
            '.main-content',
            '.content',
            '#main',
            '#content',
            '.post-content',
            '.entry-content'
        ]

        content_text = ""

        # Try each selector to find main content
        for selector in main_selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    content_text = await element.inner_text()
                    if content_text and len(content_text.strip()) > 100:
                        break
            except:
                continue

        # If no main content found, get body content and filter
        if not content_text or len(content_text.strip()) < 100:
            # Get all text content
            content_text = await page.evaluate('''
                () => {
                    // Remove script and style elements
                    const scripts = document.querySelectorAll('script, style, nav, header, footer, aside, .nav, .navigation, .menu, .sidebar');
                    scripts.forEach(el => el.remove());

                    // Get body text
                    return document.body.innerText || document.body.textContent || '';
                }
            ''')

        # Clean up the content
        content_text = clean_content(content_text)

        # Limit content length to avoid huge texts
        if len(content_text) > 5000:
            content_text = content_text[:5000] + "..."

        print("Content extracted:", content_text)
        return content_text

    except Exception as e:
        logger.error(f"Error extracting content: {e}")
        return ""

async def extract_links(browser: Browser, url: str, base_domain: str) -> List[str]:
    """
    Extract all internal links from a page.
    """
    page = await browser.new_page()
    links = []

    try:
        await page.goto(url, wait_until='domcontentloaded', timeout=20000)

        # Extract all links
        link_elements = await page.query_selector_all('a[href]')

        for link_element in link_elements:
            href = await link_element.get_attribute('href')
            if href:
                # Convert relative URLs to absolute
                absolute_url = urljoin(url, href)

                # Check if link is internal and valid
                if is_valid_internal_link(absolute_url, base_domain):
                    links.append(absolute_url)

    except Exception as e:
        logger.error(f"Error extracting links from {url}: {e}")

    finally:
        await page.close()

    print("Extracted links:", links)
    return list(set(links))  # Remove duplicates

def is_valid_internal_link(url: str, base_domain: str) -> bool:
    """
    Check if a URL is a valid internal link.
    """
    try:
        parsed = urlparse(url)

        # Must be same domain
        if parsed.netloc != base_domain:
            return False

        # Skip certain file types and paths
        excluded_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.css', '.js', '.xml', '.zip']
        excluded_paths = ['/feed/', '/rss/', '/admin/', '/wp-admin/', '/wp-content/', '/wp-includes/']

        path = parsed.path.lower()

        # Check extensions
        for ext in excluded_extensions:
            if path.endswith(ext):
                return False

        # Check paths
        for excluded_path in excluded_paths:
            if excluded_path in path:
                return False

        return True

    except Exception:
        return False

def clean_content(text: str) -> str:
    """
    Clean and normalize extracted text content.
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove common unwanted patterns
    text = re.sub(r'Skip to.*?content', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Cookie.*?policy', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Privacy.*?policy', '', text, flags=re.IGNORECASE)

    # Remove multiple consecutive periods or dashes
    text = re.sub(r'[.]{3,}', '...', text)
    text = re.sub(r'[-]{3,}', '---', text)

    return text.strip()

def generate_goals_with_ai(crawled_data: List[Dict]) -> List[str]:
    """
    Uses OpenAI's GPT model to analyze crawled website content and suggest user goals.
    """
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not set. Cannot generate AI goals.")
        return []

    if not crawled_data:
        logger.warning("No crawled data provided to generate goals.")
        return []

    logger.info("Generating user goals with AI...")

    # Combine content from the first few pages to create a context for the LLM
    # We limit this to avoid exceeding token limits
    context = ""
    for i, page in enumerate(crawled_data[:5]): # Use up to the first 5 pages
        context += f"--- Page {i+1}: {page.get('title', '')} ---\n"
        context += page.get('content', '')[:2000] # Limit content per page
        context += "\n\n"

    if not context.strip():
        logger.warning("Could not extract any content to send to AI.")
        return []

    # Define the prompt for the AI
    logger.info("Creating AI prompt for goal generation...", context)
    prompt = f"""
    You are an expert in user experience and marketing strategy. Your task is to analyze the content of a website and suggest the most common user goals or "jobs to be done".

    Based on the following content crawled from a website, identify the top 5-7 most likely actions a user would want to take on this site. These goals will be used for a navigation chatbot.

    Guidelines:
    - Each goal must be a short, clear, action-oriented phrase (e.g., "Schedule a Demo", "View Pricing", "Contact Support").
    - Do not use any introductory text like "Here are the suggested goals:".
    - Just provide a clean, new-line separated list.
    - Do not number the list or use bullet points.

    Website Content:
    {context}
    """

    try:
        # Use the ChatOpenAI model for the completion
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=OPENAI_API_KEY)
        response = llm.invoke(prompt)

        # The response content will be in response.content for Chat Models
        raw_goals = response.content.strip()

        # Process the response into a clean list
        # Split by newline and filter out any empty lines or extra characters
        goals = [
            re.sub(r'^\s*[\d\.\-\*]+\s*', '', line).strip()
            for line in raw_goals.split('\n')
            if line.strip()
        ]

        logger.info(f"AI suggested goals: {goals}")
        return goals[:8] # Return up to 8 goals

    except Exception as e:
        logger.error(f"Error calling OpenAI to generate goals: {e}")
        return []

async def cleanup_chroma_db(chroma_db_path: str, max_retries: int = 3) -> bool:
    """
    Robust ChromaDB cleanup function that tries multiple approaches.
    Returns True if successful, False otherwise.
    """
    global chroma_client
    
    # Close any existing connections
    if chroma_client:
        try:
            if hasattr(chroma_client, 'close'):
                chroma_client.close()
        except:
            pass
        chroma_client = None
    
    gc.collect()
    
    path = Path(chroma_db_path)
    if not path.exists():
        return True
    
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                # Direct removal
                shutil.rmtree(path)
                return True
            elif attempt == 1:
                # Individual file removal
                for file_path in path.rglob("*"):
                    if file_path.is_file():
                        file_path.unlink(missing_ok=True)
                for dir_path in sorted(path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                    if dir_path.is_dir():
                        try:
                            dir_path.rmdir()
                        except OSError:
                            pass
                if path.exists():
                    path.rmdir()
                return True
            else:
                # Rename method
                backup_name = f"{chroma_db_path}_backup_{int(time.time())}"
                path.rename(backup_name)
                path.mkdir(parents=True, exist_ok=True)
                return True
                
        except Exception as e:
            logger.warning(f"ChromaDB cleanup attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                gc.collect()
    
    return False
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
    """
    Generate sitemap using Playwright and suggest user goals using AI.
    """
    domain = request.domain
    logger.info(f"Starting sitemap and goal generation for domain: {domain}")

    try:
        # await clear_knowledge_base()
        logger.info("Cleared previous knowledge base to prepare for new crawl data.")
        # Step 1: Crawl the website using Playwright
        sitemap_data = await crawl_with_playwright(domain, max_pages=110)
        logger.info(f"Finished crawling. Found {len(sitemap_data)} pages.")

        if not sitemap_data:
            # More specific error handling
            logger.warning(f"No pages found for domain: {domain}")

            # Try to provide more helpful error information
            if not domain.startswith(('http://', 'https://')):
                test_url = f'https://{domain}'
            else:
                test_url = domain

            raise HTTPException(
                status_code=404,
                detail=f"No pages found for {domain}. Please check if the website is accessible at {test_url} and allows crawling."
            )
            
        # Step 2: Generate goal suggestions using AI
        logger.info("data ----------------------------", sitemap_data)
        chunks_added_to_rag = add_website_content_to_rag(sitemap_data)
        if chunks_added_to_rag > 0:
            logger.info(f"Successfully added {chunks_added_to_rag} content chunks to the RAG vector store.")
        else:
            logger.warning("Could not add any website content to the RAG system.")
        suggested_goals = generate_goals_with_ai(sitemap_data)
        if not suggested_goals:
            logger.warning("AI goal generation failed. Falling back to keyword-based suggestions.")
            suggested_goals = suggest_goals_from_content(sitemap_data)
        logger.info(f"Final suggested goals: {suggested_goals}")
        
        # Fallback to keyword-based suggestions if AI fails or returns nothing
        if not suggested_goals:
            logger.warning("AI goal generation failed or returned no goals. Falling back to keyword-based suggestions.")
            suggested_goals = suggest_goals_from_content(sitemap_data)

        logger.info(f"Final suggested goals: {suggested_goals}")

        # Step 3: Save the sitemap to the database
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sitemap")  # Clear old sitemap
            inserted_count = 0
            for page in sitemap_data:
                try:
                    cursor.execute(
                        "INSERT INTO sitemap (url, title, content, parent_url, depth) VALUES (?, ?, ?, ?, ?)",
                        (page['url'], page['title'], page['content'],
                         page.get('parent_url'), page.get('depth', 0))
                    )
                    inserted_count += 1
                except Exception as db_error:
                    logger.warning(f"Failed to insert page {page.get('url', 'unknown')}: {db_error}")
                    continue

            conn.commit()
            logger.info(f"Inserted {inserted_count} new sitemap entries into DB.")

        return {
            "message": "Sitemap generated and goals suggested successfully",
            "pages_found": len(sitemap_data),
            "pages_saved_to_db": inserted_count,
            "chunks_added_to_rag": chunks_added_to_rag, 
            "suggested_goals": suggested_goals
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error generating sitemap for {domain}: {e}", exc_info=True)
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
    domain = verification.domain.replace('http://', '').replace('https://', '').split('/')[0] 
    logger.info(f"Attempting to verify domain: {domain}")
    try:
        response = requests.head(f"https://{domain}", timeout=15, verify=True)

        if 200 <= response.status_code < 400:
            with get_db_connection() as conn:
                cursor = conn.cursor()
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
                background: linear-gradient(135deg, #000412 0%, #000412 100%);
                color: #333;
            }

            .header {
                background: rgba(255, 255, 255, 0.01);
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
                color: white;
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
                background: rgba(255, 255, 255, 0.01);
                backdrop-filter: blur(10px);
                padding: 16px 20px;
                border-top: 1px solid rgba(0, 0, 0, 0.1);
                display: flex;
                flex-direction: row;
                align-items: center;
            }

            .input-wrapper {
               display: flex;
                flex: 1;
                gap: 12px;
                align-items: center;
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
                background: rgb(229, 222, 222);
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

        <div class="input-container ">
            <div class="input-wrapper">
                <textarea
                    id="messageInput"
                    class="message-input"
                    placeholder="Type your message..."
                    rows="1"
                ></textarea>
                <button class="send-button" id="sendButton" onclick="sendMessage()">
                    <svg viewBox="0 0 24 24" fill="currentColor">
                        <path d="M2,21L23,12L2,3V10L17,12L2,14V21Z" />
                    </svg>
                </button>
            </div>
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
    logger.info("Received GET request for /api/sitemap")
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT url, title, depth FROM sitemap ORDER BY depth, url")
            sitemap = [dict(row) for row in cursor.fetchall()]
        logger.info(f"Returning {len(sitemap)} sitemap entries.")
        return {"sitemap": sitemap}
    except Exception as e:
         logger.error(f"Error fetching sitemap: {e}") 
         raise HTTPException(status_code=500, detail=f"Error fetching sitemap: {e}")

@app.delete("/api/knowledge-base")
async def clear_knowledge_base():
    """Deletes all documents metadata from SQLite and clears Chroma DB."""
    global rag_chain, chroma_client
    
    logger.info("Received DELETE request for /api/knowledge-base")
    
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG system is not initialized. Cannot clear knowledge base.")

    try:
        # Step 1: Properly close existing RAG system and ChromaDB connections
        logger.info("Shutting down existing RAG system connections...")
        
        # Close ChromaDB client if it exists
        if chroma_client:
            try:
                if hasattr(chroma_client, 'close'):
                    chroma_client.close()
                elif hasattr(chroma_client, 'reset'):
                    chroma_client.reset()
                logger.info("Closed ChromaDB client connection")
            except Exception as e:
                logger.warning(f"Error closing ChromaDB client: {e}")
            finally:
                chroma_client = None
        
        # Reset RAG chain reference
        rag_chain = None
        
        # Force garbage collection to release file handles
        gc.collect()
        
        # Small delay to ensure file handles are released
        await asyncio.sleep(0.1)

        # Step 2: Clear SQLite metadata
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM knowledge_base")
            conn.commit()
            logger.info("Cleared knowledge_base table in SQLite.")

        # Step 3: Clear Chroma DB directory with robust error handling
        chroma_db_path = Path(CHROMA_DB_PATH)
        
        if chroma_db_path.exists():
            logger.info(f"Clearing Chroma DB directory: {CHROMA_DB_PATH}")
            
            success = False
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    # Method 1: Try direct removal
                    if attempt == 0:
                        shutil.rmtree(chroma_db_path)
                        success = True
                        logger.info("Successfully removed ChromaDB directory (direct method)")
                        break
                    
                    # Method 2: Remove files individually then directory
                    elif attempt == 1:
                        logger.info(f"Attempt {attempt + 1}: Removing files individually...")
                        
                        # Remove all files first
                        for file_path in chroma_db_path.rglob("*"):
                            if file_path.is_file():
                                try:
                                    file_path.unlink()
                                except (PermissionError, OSError) as e:
                                    logger.warning(f"Could not delete file {file_path}: {e}")
                        
                        # Remove empty directories
                        for dir_path in sorted(chroma_db_path.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                            if dir_path.is_dir():
                                try:
                                    dir_path.rmdir()
                                except OSError:
                                    pass  # Directory not empty, skip
                        
                        # Remove main directory
                        if chroma_db_path.exists():
                            chroma_db_path.rmdir()
                        
                        success = True
                        logger.info("Successfully removed ChromaDB directory (individual file method)")
                        break
                    
                    # Method 3: Rename and create new directory
                    else:
                        logger.info(f"Attempt {attempt + 1}: Using rename method as fallback...")
                        backup_name = f"{CHROMA_DB_PATH}_backup_{int(time.time())}"
                        backup_path = Path(backup_name)
                        
                        chroma_db_path.rename(backup_path)
                        logger.info(f"Renamed old ChromaDB directory to {backup_name}")
                        
                        # Create new empty directory
                        chroma_db_path.mkdir(parents=True, exist_ok=True)
                        success = True
                        logger.info("Created new empty ChromaDB directory")
                        break
                
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        # Wait before retry
                        await asyncio.sleep(1)
                        gc.collect()  # Try to free up resources
                    else:
                        logger.error(f"All {max_retries} attempts to clear ChromaDB failed")
                        # Don't raise exception, just log and continue
            
            if not success:
                logger.warning("Could not fully clear ChromaDB directory, but continuing with re-initialization")
        else:
            logger.warning(f"Chroma DB directory not found at {CHROMA_DB_PATH}. Nothing to clear.")

        # Step 4: Re-initialize RAG system
        try:
            # Ensure the ChromaDB directory exists
            chroma_db_path.mkdir(parents=True, exist_ok=True)
            
            # Re-initialize RAG system
            initialize_rag_system()
            logger.info("RAG system re-initialized successfully.")
            
            return {"message": "Knowledge base cleared and RAG system re-initialized successfully."}
            
        except Exception as init_error:
            logger.error(f"Error re-initializing RAG system: {init_error}")
            return {
                "message": "Knowledge base cleared but RAG system re-initialization failed. Please restart the application.",
                "warning": str(init_error)
            }

    except sqlite3.Error as db_error:
        logger.error(f"Database error clearing knowledge base: {db_error}")
        raise HTTPException(status_code=500, detail=f"Database error clearing knowledge base: {db_error}")
    except Exception as e:
        logger.error(f"Error clearing knowledge base: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.get("/api/status")
async def get_status():
    """Provides status of the RAG system and database."""
    logger.info("Received GET request for /api/status")
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
    import os
    PORT = int(os.environ.get("PORT", 8000))
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
