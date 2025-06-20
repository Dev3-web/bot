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
import logging as logger # Using logger for consistency
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
import asyncio
import sys
import gc
from pathlib import Path
import aiohttp

# --- Langchain / RAG Imports ---
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

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

if sys.platform == "win32":
    # Use ProactorEventLoopPolicy on Windows for better async compatibility
    # This must be set before any asyncio loop is created, ideally at the very beginning
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

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
            description TEXT,           -- New field
            keywords TEXT,              -- New field
            og_title TEXT,              -- New field
            og_description TEXT,        -- New field
            og_image TEXT,              -- New field
            publication_date TIMESTAMP, -- New field
            created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Add new columns if they don't exist (for existing databases)
    def add_column_if_not_exists(table_name, column_name, column_type):
        try:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
            logger.info(f"Added column '{column_name}' to table '{table_name}'.")
        except sqlite3.OperationalError as e:
            if f"duplicate column name: {column_name}" in str(e):
                logger.debug(f"Column '{column_name}' already exists in table '{table_name}'.")
            else:
                logger.error(f"Error adding column '{column_name}' to table '{table_name}': {e}")

    add_column_if_not_exists('sitemap', 'description', 'TEXT')
    add_column_if_not_exists('sitemap', 'keywords', 'TEXT')
    add_column_if_not_exists('sitemap', 'og_title', 'TEXT')
    add_column_if_not_exists('sitemap', 'og_description', 'TEXT')
    add_column_if_not_exists('sitemap', 'og_image', 'TEXT')
    add_column_if_not_exists('sitemap', 'publication_date', 'TIMESTAMP')


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
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

        logger.info(f"Adding {len(split_docs)} chunks to Chroma DB...")
        vectorstore.add_documents(split_docs)
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
                    "title": page.get('title', 'Untitled'),
                    "description": page.get('description'),
                    "og_title": page.get('og_title'),
                    "og_description": page.get('og_description'),
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

async def crawl_website_aiohttp(url: str, max_pages: int = 110, max_concurrent: int = 10) -> List[Dict]:
    """
    Crawls website using aiohttp for Windows compatibility.
    """
    normalized_url = url if url.startswith(('http://', 'https://')) else f'https://{url}'
    base_domain = urlparse(normalized_url).netloc

    logger.info(f"Starting aiohttp-based crawl for: {normalized_url} (max_concurrent: {max_concurrent})")

    discovered_pages = []
    urls_to_visit = deque([normalized_url])
    visited_urls = {normalized_url}

    # Create aiohttp session with proper configuration
    timeout = aiohttp.ClientTimeout(total=30, connect=10)
    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)

    async with aiohttp.ClientSession(
        timeout=timeout,
        connector=connector,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    ) as session:

        while urls_to_visit and len(discovered_pages) < max_pages:
            # Create batch of URLs to process
            batch_urls = []
            batch_size = min(len(urls_to_visit), max_concurrent, max_pages - len(discovered_pages))

            for _ in range(batch_size):
                if urls_to_visit:
                    batch_urls.append(urls_to_visit.popleft())

            if not batch_urls:
                break # No more URLs to process in this iteration

            # Process batch concurrently
            tasks = [fetch_page_content(session, url, base_domain) for url in batch_urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Error during page fetch: {result}")
                    continue

                if result:
                    page_data, new_links = result
                    if page_data:
                        discovered_pages.append(page_data)

                        # Add new URLs to visit
                        for link in new_links:
                            # Basic URL normalization (remove query params and fragments) for deduplication
                            parsed_link = urlparse(link)
                            clean_link = urlunparse(parsed_link._replace(query='', fragment=''))
                            if clean_link not in visited_urls and len(discovered_pages) < max_pages:
                                visited_urls.add(clean_link)
                                urls_to_visit.append(clean_link)

            # Small delay to be respectful to the server
            await asyncio.sleep(0.1)

    logger.info(f"aiohttp crawling finished. Collected {len(discovered_pages)} pages.")
    return discovered_pages


async def fetch_page_content(session: aiohttp.ClientSession, url: str, base_domain: str) -> Optional[tuple]:
    """
    Fetches a single page content and extracts links and richer metadata.
    """
    try:
        async with session.get(url) as response:
            if response.status != 200:
                logger.warning(f"HTTP {response.status} for {url}")
                return None

            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type and not content_type.startswith('text/html'):
                logger.warning(f"Skipping non-HTML content: {url} (Content-Type: {content_type})")
                return None

            html_content = await response.text(encoding='utf-8', errors='ignore')

            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')

            # --- Extract Metadata ---
            page_title = (soup.find('title').get_text().strip() if soup.find('title') else
                          soup.find('meta', property='og:title').get('content', '').strip() if soup.find('meta', property='og:title') else
                          'Untitled')

            page_description = (soup.find('meta', attrs={'name': 'description'}).get('content', '').strip()
                                if soup.find('meta', attrs={'name': 'description'}) else
                                soup.find('meta', property='og:description').get('content', '').strip()
                                if soup.find('meta', property='og:description') else None)

            page_keywords = (soup.find('meta', attrs={'name': 'keywords'}).get('content', '').strip()
                             if soup.find('meta', attrs={'name': 'keywords'}) else None)

            og_title = soup.find('meta', property='og:title').get('content', '').strip() if soup.find('meta', property='og:title') else None
            og_description = soup.find('meta', property='og:description').get('content', '').strip() if soup.find('meta', property='og:description') else None
            og_image = soup.find('meta', property='og:image').get('content', '').strip() if soup.find('meta', property='og:image') else None

            publication_date = None
            time_tag = soup.find('time', datetime=True)
            if time_tag:
                try:
                    publication_date = datetime.fromisoformat(time_tag['datetime']).isoformat()
                except ValueError:
                    pass # Invalid datetime format

            # --- Extract Main Content ---
            # Remove script and style elements, and common boilerplate sections
            for script in soup(["script", "style", "nav", "header", "footer", "aside", ".sidebar", ".ad", ".social-share", ".wp-block-comments", ".comments", "#comments"]):
                script.decompose()

            # Prioritize main content areas
            main_content_elements = soup.find_all(['main', 'article', 'div'], class_=['content', 'post-content', 'entry-content', 'main-content'])
            page_content = ""
            for elem in main_content_elements:
                text = elem.get_text(separator=' ', strip=True)
                if len(text) > 100: # Only use if it seems like substantial content
                    page_content = text
                    break
            
            # Fallback to body content if specific main content wasn't found or was too short
            if not page_content:
                page_content = soup.get_text(separator=' ', strip=True)

            # Clean up text
            page_content = re.sub(r'\s+', ' ', page_content).strip()[:5000] # Limit content length for RAG

            # --- Extract Internal Links ---
            internal_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)
                parsed_url = urlparse(full_url)

                # Skip non-http/https links, mailto, tel etc.
                if parsed_url.scheme not in ('http', 'https'):
                    continue

                # Check if it's an internal link and not a file download
                if parsed_url.netloc == base_domain and not re.search(r'\.(pdf|docx|xlsx|pptx|zip|rar|exe|jpg|png|gif|svg)$', parsed_url.path.lower()):
                    # Clean URL (remove fragments and query params for deduplication)
                    clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
                    if clean_url not in internal_links and clean_url != url: # Avoid self-referencing
                        internal_links.append(clean_url)

            page_data = {
                'url': url,
                'title': page_title,
                'content': page_content,
                'depth': urlparse(url).path.count('/'),
                'parent_url': None, # This could be tracked if desired in a more complex crawler
                'description': page_description,
                'keywords': page_keywords,
                'og_title': og_title,
                'og_description': og_description,
                'og_image': og_image,
                'publication_date': publication_date,
            }

            return page_data, internal_links[:20] # Limit links to avoid excessive queueing

    except asyncio.TimeoutError:
        logger.warning(f"Timeout fetching {url}")
        return None
    except Exception as e:
        logger.warning(f"Error fetching {url}: {e}")
        return None


async def save_sitemap_to_database(sitemap_data: List[Dict]) -> int:
    """
    Saves sitemap data to database with improved error handling.
    """
    saved_count = 0

    # Run database operations in a thread to avoid blocking
    def save_to_db():
        nonlocal saved_count
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()

                # Clear existing sitemap data
                cursor.execute("DELETE FROM sitemap")
                logger.info("Cleared existing sitemap data.")

                # Insert new data
                for page_info in sitemap_data:
                    try:
                        cursor.execute(
                            """INSERT INTO sitemap (
                                url, title, content, parent_url, depth,
                                description, keywords, og_title, og_description, og_image, publication_date
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                page_info['url'],
                                page_info['title'],
                                page_info['content'],
                                page_info.get('parent_url'),
                                page_info.get('depth', 0),
                                page_info.get('description'),
                                page_info.get('keywords'),
                                page_info.get('og_title'),
                                page_info.get('og_description'),
                                page_info.get('og_image'),
                                page_info.get('publication_date')
                            )
                        )
                        saved_count += 1

                    except Exception as db_error:
                        logger.warning(f"Database insert failed for {page_info.get('url', 'unknown')}: {db_error}")
                        continue

                conn.commit()
                logger.info(f"Database transaction committed. {saved_count} records saved.")
        except Exception as e:
            logger.error(f"Database operation failed: {e}")

    # Run in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, save_to_db)

    return saved_count


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
    # Use first 5 pages for context, prioritize pages with content
    relevant_pages = [p for p in crawled_data if p.get('content')]
    for i, page in enumerate(relevant_pages[:5]): # Use up to the first 5 relevant pages
        context += f"--- Page {i+1}: {page.get('title', '')} ---\n"
        context += page.get('content', '')[:2000] # Limit content per page
        context += "\n\n"

    if not context.strip():
        logger.warning("Could not extract any content to send to AI for goal generation.")
        return []

    # Define the prompt for the AI
    logger.debug(f"Creating AI prompt for goal generation with context length: {len(context)}")
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

def generate_test_scenarios_with_ai(crawled_data: List[Dict]) -> List[str]:
    """
    Uses OpenAI's GPT model to analyze website content and suggest test scenarios 
    (i.e., sample user questions to simulate real usage).
    """
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not set. Cannot generate AI test scenarios.")
        return []

    if not crawled_data:
        logger.warning("No crawled data provided to generate test scenarios.")
        return []

    logger.info("Generating test scenarios with AI...")

    # Extract content for context
    context = ""
    relevant_pages = [p for p in crawled_data if p.get('content')]
    for i, page in enumerate(relevant_pages[:5]):
        context += f"--- Page {i+1}: {page.get('title', '')} ---\n"
        context += page.get('content', '')[:2000]
        context += "\n\n"

    if not context.strip():
        logger.warning("No valid content for test scenario generation.")
        return []

    # Prompt for AI
    prompt = f"""
    You are a product experience strategist helping to simulate how users might interact with a chatbot on a website.

    Based on the following website content, generate a list of realistic test questions or help-seeking scenarios a user might ask to test the system.

    Guidelines:
    - Each scenario should be short (preferably one sentence) and represent a user question or intent (e.g., "How do I reset my password?", "Can you show me the pricing options?")
    - Avoid bullet points or numbers.
    - Just provide a clean, new-line-separated list.

    Website Content:
    {context}
    """

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=OPENAI_API_KEY)
        response = llm.invoke(prompt)

        raw_scenarios = response.content.strip()
        scenarios = [
            re.sub(r'^\s*[\d\.\-\*]+\s*', '', line).strip()
            for line in raw_scenarios.split('\n')
            if line.strip()
        ]

        logger.info(f"AI suggested test scenarios: {scenarios}")
        return scenarios[:8]  # Limit to top 8 test cases

    except Exception as e:
        logger.error(f"Error generating AI test scenarios: {e}")
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
    Generate sitemap using aiohttp-based crawler (Windows-compatible).
    """
    domain = request.domain
    logger.info(f"Initiating sitemap generation for domain: {domain}")

    try:
        # Clear previous data
        logger.info("Preparing knowledge base for new crawl session.")

        # Step 1: Crawl website using aiohttp approach
        crawl_results = await crawl_website_aiohttp(domain, max_pages=110)
        logger.info(f"Crawl completed. Retrieved {len(crawl_results)} pages.")

        if not crawl_results:
            logger.warning(f"Zero pages discovered for domain: {domain}")

            # Construct test URL for better error messaging
            test_domain = domain if domain.startswith(('http://', 'https://')) else f'https://{domain}'

            raise HTTPException(
                status_code=404,
                detail=f"Unable to crawl {domain}. Verify the website is reachable at {test_domain} and permits crawling."
            )

        # Step 2: Process content for RAG system
        logger.info("Processing crawled content for RAG integration.")
        rag_chunks = add_website_content_to_rag(crawl_results)

        if rag_chunks > 0:
            logger.info(f"Added {rag_chunks} content chunks to RAG vector store.")
        else:
            logger.warning("RAG integration failed - no content chunks added.")

        # Step 3: Generate AI-powered goal suggestions
        logger.info("Generating AI-powered goal suggestions.")
        ai_goals = generate_goals_with_ai(crawl_results)

        # Fallback to keyword-based approach if AI fails
        final_goals = ai_goals if ai_goals else suggest_goals_from_content(crawl_results)

        if not final_goals:
            logger.warning("Both AI and keyword-based goal generation failed.")
            final_goals = ["Improve website engagement", "Enhance user experience"]  # Default goals

        logger.info(f"Generated goals: {final_goals}")

        # Step 4: Persist sitemap data to database
        db_records_saved = await save_sitemap_to_database(crawl_results)
        logger.info(f"Persisted {db_records_saved} sitemap records to database.")

        return {
            "status": "success",
            "message": "Sitemap generation completed successfully",
            "metrics": {
                "pages_discovered": len(crawl_results),
                "pages_persisted": db_records_saved,
                "rag_chunks_added": rag_chunks
            },
            "suggested_goals": final_goals
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sitemap generation failed for {domain}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error during sitemap generation: {str(e)}"
        )

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
            cursor.execute("SELECT url, title, depth, description, og_title, og_image, publication_date FROM sitemap ORDER BY depth, url")
            sitemap = [dict(row) for row in cursor.fetchall()]
        logger.info(f"Returning {len(sitemap)} sitemap entries.")
        return {"sitemap": sitemap}
    except Exception as e:
         logger.error(f"Error fetching sitemap: {e}")
         raise HTTPException(status_code=500, detail=f"Error fetching sitemap: {e}")

@app.delete("/api/knowledge-base/document/{filename:path}")
async def delete_document_from_knowledge_base(filename: str):
    decoded_filename = unquote(filename)
    logger.info(f"Received DELETE request for document: {decoded_filename}")

    if not rag_chain or not hasattr(rag_chain, 'retriever') or \
       not hasattr(rag_chain.retriever, 'vectorstore') or \
       not isinstance(rag_chain.retriever.vectorstore, Chroma):
        logger.error("RAG system or Chroma vector store not properly initialized for deletion.")
        raise HTTPException(status_code=500, detail="RAG system/vector store not available.")

    vectorstore = rag_chain.retriever.vectorstore
    rows_deleted_sqlite = 0
    chroma_chunks_deleted_count = 0
    doc_ids_to_delete = []

    # Step 1: Try to find document IDs in Chroma first (before deleting from SQLite)
    # This is because 'source' metadata comes from the original filename.
    try:
        # Langchain's Chroma get() method can filter by metadata.
        # It returns a dict with 'ids', 'embeddings', 'metadatas', 'documents'.
        retrieved_docs_info = vectorstore.get(where={"source": decoded_filename})
        doc_ids_to_delete = retrieved_docs_info.get('ids', [])

        if not doc_ids_to_delete:
            logger.info(f"No document chunks found in Chroma with source '{decoded_filename}'.")
        else:
            logger.info(f"Found {len(doc_ids_to_delete)} document chunks in Chroma for '{decoded_filename}'.")

    except Exception as e:
        logger.error(f"Error querying Chroma for document chunks of '{decoded_filename}': {e}", exc_info=True)
        # Proceed to SQLite deletion attempt, but log this issue.
        # No HTTP Exception here yet, as SQLite might still succeed or also find nothing.

    # Step 2: Delete from SQLite
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Important: Get the ID or other info IF NEEDED before deleting
            # For now, filename is the key.
            cursor.execute("DELETE FROM knowledge_base WHERE filename = ?", (decoded_filename,))
            rows_deleted_sqlite = cursor.rowcount
            conn.commit()
        
        if rows_deleted_sqlite == 0:
            logger.warning(f"Document '{decoded_filename}' not found in SQLite knowledge_base table.")
            # If also not found in Chroma, then it's a true 404
            if not doc_ids_to_delete:
                raise HTTPException(status_code=404, detail=f"Document '{decoded_filename}' not found.")
        else:
            logger.info(f"Successfully deleted '{decoded_filename}' metadata from SQLite.")

    except HTTPException as http_exc: # Propagate 404 if raised
        raise http_exc
    except sqlite3.Error as e:
        logger.error(f"SQLite error deleting document metadata for '{decoded_filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Database error deleting document metadata: {str(e)}")

    # Step 3: Delete from Chroma if IDs were found
    if doc_ids_to_delete:
        try:
            vectorstore.delete(ids=doc_ids_to_delete)
            chroma_chunks_deleted_count = len(doc_ids_to_delete)
            logger.info(f"Successfully deleted {chroma_chunks_deleted_count} chunks for '{decoded_filename}' from Chroma.")
        except Exception as e:
            logger.error(f"Chroma DB error deleting document chunks for '{decoded_filename}': {e}", exc_info=True)
            # If SQLite deletion succeeded but Chroma failed, this is a partial failure.
            if rows_deleted_sqlite > 0:
                 return JSONResponse(
                    status_code=207, # Multi-Status
                    content={
                        "message": f"Successfully deleted '{decoded_filename}' from database metadata, "
                                   f"but failed to delete {len(doc_ids_to_delete)} associated data chunks from vector store. "
                                   f"Please check logs. Error: {str(e)}",
                        "sqlite_deleted": rows_deleted_sqlite > 0,
                        "chroma_expected_deletions": len(doc_ids_to_delete),
                        "chroma_actual_deletions": 0
                    }
                )
            # If SQLite also failed to find it, but Chroma did (should be rare if consistent), still raise error
            raise HTTPException(status_code=500, detail=f"Vector store error deleting document data: {str(e)}")

    # Final success message
    return {
        "message": f"Document '{decoded_filename}' and its associated data successfully processed for deletion.",
        "sqlite_deleted_metadata_rows": rows_deleted_sqlite,
        "chroma_deleted_chunks": chroma_chunks_deleted_count
    }

@app.delete("/api/knowledge-base")
async def clear_knowledge_base():
    global rag_chain, chroma_client
    logger.info("Received DELETE request for /api/knowledge-base (full clear)")
    if not rag_chain:
        logger.warning("RAG system not initialized. Attempting cleanup anyway.")
    
    # Close ChromaDB client if it exists (Langchain's Chroma wrapper handles its own client)
    # The `chroma_client` global might be for a direct `chromadb.Client` instance
    # if you were using one elsewhere, which you are not explicitly here for RAG.
    # So, this part might be less relevant unless `chroma_client` is used by `cleanup_chroma_db`.
    if 'chroma_client' in globals() and chroma_client: # Check if chroma_client is defined and not None
        try:
            if hasattr(chroma_client, 'close'): chroma_client.close()
            elif hasattr(chroma_client, 'reset'): chroma_client.reset() # Some versions might have reset
            logger.info("Closed direct ChromaDB client connection if it existed.")
        except Exception as e:
            logger.warning(f"Error closing direct ChromaDB client: {e}")
        finally:
            chroma_client = None

    # Reset RAG chain reference. This effectively "disconnects" it from the old vector store.
    # The vector store itself (files on disk) needs separate handling.
    rag_chain = None 
    gc.collect()
    await asyncio.sleep(0.1) # Give a moment for resources to release

    # Clear SQLite metadata
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM knowledge_base")
            conn.commit()
            logger.info("Cleared knowledge_base table in SQLite.")
    except sqlite3.Error as db_error:
        logger.error(f"Database error clearing knowledge base table: {db_error}")
        # Don't raise immediately, try to clear Chroma too.

    # Clear Chroma DB directory using the robust cleanup function
    chroma_db_cleared_successfully = await cleanup_chroma_db(CHROMA_DB_PATH)
    if chroma_db_cleared_successfully:
        logger.info(f"Successfully cleared Chroma DB directory: {CHROMA_DB_PATH}")
    else:
        logger.error(f"Failed to fully clear Chroma DB directory: {CHROMA_DB_PATH}. Manual cleanup might be needed.")

    # Re-initialize RAG system to create a fresh, empty vector store
    try:
        Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True) # Ensure dir exists for new init
        initialize_rag_system()
        logger.info("RAG system re-initialized successfully after clearing.")
        return {"message": "Knowledge base cleared and RAG system re-initialized successfully."}
    except Exception as init_error:
        logger.error(f"Error re-initializing RAG system after clearing: {init_error}", exc_info=True)
        return JSONResponse(
            status_code=500, # Internal Server Error
            content={
                "message": "Knowledge base tables cleared, and ChromaDB directory removed (if possible), "
                           "but RAG system re-initialization failed. Please check logs and restart the application.",
                "warning": str(init_error),
                "chroma_dir_cleared": chroma_db_cleared_successfully
            }
        )

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
            "knowledge_base_metadata": kb_count,
            "knowledge_base_chunks_in_chroma": chroma_count,
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