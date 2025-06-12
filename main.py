#!/usr/bin/env python3
"""
Navigation Helper Bot - Backend API
Complete backend system for the navigation helper bot with all features
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Navigation Helper Bot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
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
            content TEXT NOT NULL,
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
            domain TEXT NOT NULL,
            widget_position TEXT DEFAULT 'bottom-right',
            widget_color TEXT DEFAULT '#007bff',
            goals TEXT,
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
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# Pydantic models
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
    goals: List[str] = []

class DNSVerification(BaseModel):
    domain: str

class WidgetRequest(BaseModel):
    domain: str

# Helper functions
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        return ""

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(io.BytesIO(file_content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {e}")
        return ""

def generate_file_hash(content: bytes) -> str:
    """Generate SHA-256 hash of file content"""
    return hashlib.sha256(content).hexdigest()

def crawl_website(base_url: str, max_depth: int = 3) -> List[Dict]:
    """Crawl website and extract sitemap information"""
    visited = set()
    sitemap_data = []
    
    def crawl_page(url: str, depth: int = 0, parent_url: str = None):
        if depth > max_depth or url in visited:
            return
        
        visited.add(url)
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Extract main content
            content = ""
            for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                content += tag.get_text() + " "
            
            sitemap_data.append({
                'url': url,
                'title': title_text,
                'content': content.strip()[:1000],  # Limit content length
                'parent_url': parent_url,
                'depth': depth
            })
            
            # Find all links for further crawling
            if depth < max_depth:
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    full_url = urljoin(url, href)
                    
                    # Only crawl internal links
                    if urlparse(full_url).netloc == urlparse(base_url).netloc:
                        crawl_page(full_url, depth + 1, url)
                        
        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
    
    crawl_page(base_url)
    return sitemap_data

def generate_bot_response(user_message: str, session_id: str) -> str:
    """Generate bot response based on knowledge base and user goals"""
    user_message_lower = user_message.lower()
    
    # Check if user is asking for help with specific goals
    if any(keyword in user_message_lower for keyword in ['upload', 'document', 'file']):
        return """I can help you upload documents! Here's how:
        
1. Go to the 'Upload Documents' section
2. Select your files (PDF, DOCX, TXT supported)
3. Click 'Upload' to add them to the knowledge base
4. Once uploaded, you can ask questions about your documents

Would you like me to guide you to the upload section?"""
    
    elif any(keyword in user_message_lower for keyword in ['question', 'ask', 'q&a', 'query']):
        return """Great! You can use our Q&A feature to interact with uploaded documents:

1. Make sure you've uploaded your documents first
2. Type your questions in natural language
3. I'll search through your documents to find relevant answers
4. You can ask follow-up questions for more details

What would you like to know about your documents?"""
    
    elif any(keyword in user_message_lower for keyword in ['contact', 'email', 'phone', 'info']):
        return """I'd be happy to help you get in touch! Could you please provide:

1. Your name
2. Email address  
3. Phone number (optional)
4. How can we assist you?

This will help us provide better support."""
    
    else:
        # Search knowledge base for relevant content
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT content FROM knowledge_base WHERE content LIKE ?",
                (f"%{user_message}%",)
            )
            results = cursor.fetchone()
            
            if results:
                return f"Based on your documents: {results['content'][:200]}..."
            else:
                return """I'm here to help you navigate our platform! I can assist with:

• Uploading and managing documents
• Using the Q&A feature
• Collecting your contact information
• Guiding you through platform features

What would you like help with today?"""

# API Routes
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_database()

@app.get("/")
async def root():
    return {"message": "Navigation Helper Bot API is running"}

@app.post("/api/upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents to knowledge base"""
    uploaded_files = []
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        for file in files:
            try:
                content = await file.read()
                file_hash = generate_file_hash(content)
                
                # Check if file already exists
                cursor.execute("SELECT id FROM knowledge_base WHERE file_hash = ?", (file_hash,))
                if cursor.fetchone():
                    continue  # Skip duplicate files
                
                # Extract text based on file type
                if file.filename.lower().endswith('.pdf'):
                    text_content = extract_text_from_pdf(content)
                    file_type = 'pdf'
                elif file.filename.lower().endswith('.docx'):
                    text_content = extract_text_from_docx(content)
                    file_type = 'docx'
                elif file.filename.lower().endswith('.txt'):
                    text_content = content.decode('utf-8')
                    file_type = 'txt'
                else:
                    continue  # Skip unsupported file types
                
                # Insert into database
                cursor.execute("""
                    INSERT INTO knowledge_base (filename, content, file_type, file_hash)
                    VALUES (?, ?, ?, ?)
                """, (file.filename, text_content, file_type, file_hash))
                
                uploaded_files.append({
                    'filename': file.filename,
                    'type': file_type,
                    'size': len(content)
                })
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {e}")
                continue
        
        conn.commit()
    
    return {"uploaded_files": uploaded_files, "count": len(uploaded_files)}

class SitemapRequest(BaseModel):
    domain: str

@app.post("/api/generate-sitemap")
async def generate_sitemap(request: SitemapRequest):
    """Generate sitemap by crawling the website"""
    domain = request.domain
    try:
        if not domain.startswith(('http://', 'https://')):
            domain = 'https://' + domain
        
        sitemap_data = crawl_website(domain)
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Clear existing sitemap data
            cursor.execute("DELETE FROM sitemap")
            
            # Insert new sitemap data
            for page in sitemap_data:
                cursor.execute("""
                    INSERT INTO sitemap (url, title, content, parent_url, depth)
                    VALUES (?, ?, ?, ?, ?)
                """, (page['url'], page['title'], page['content'], 
                     page['parent_url'], page['depth']))
            
            conn.commit()
        
        return {"message": "Sitemap generated successfully", "pages_found": len(sitemap_data)}
        
    except Exception as e:
        logger.error(f"Error generating sitemap: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat(message: ChatMessage):
    """Handle chat messages"""
    try:
        response = generate_bot_response(message.message, message.session_id)
        
        # Log the conversation
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chat_logs (session_id, user_message, bot_response)
                VALUES (?, ?, ?)
            """, (message.session_id, message.message, response))
            conn.commit()
        
        return {"response": response}
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/collect-customer-info")
async def collect_customer_info(info: CustomerInfo):
    """Collect customer information"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO customer_info (session_id, name, email, phone, additional_info)
                VALUES (?, ?, ?, ?, ?)
            """, (info.session_id, info.name, info.email, info.phone, info.additional_info))
            conn.commit()
        
        return {"message": "Customer information collected successfully"}
        
    except Exception as e:
        logger.error(f"Error collecting customer info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/configure-bot")
async def configure_bot(config: BotConfig):
    """Configure bot settings"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO bot_config (domain, widget_position, widget_color, goals)
                VALUES (?, ?, ?, ?)
            """, (config.domain, config.widget_position, config.widget_color, 
                 json.dumps(config.goals)))
            conn.commit()
        
        return {"message": "Bot configuration saved successfully"}
        
    except Exception as e:
        logger.error(f"Error configuring bot: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/verify-dns")
async def verify_dns(verification: DNSVerification):
    """Verify DNS record for domain"""
    try:
        # Simple DNS verification - in production, implement proper DNS record checking
        domain = verification.domain.replace('http://', '').replace('https://', '')
        
        # Try to access the domain
        response = requests.get(f"https://{domain}", timeout=10)
        
        if response.status_code == 200:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE bot_config SET dns_verified = TRUE WHERE domain = ?
                """, (domain,))
                conn.commit()
            
            return {"verified": True, "message": "Domain verified successfully"}
        else:
            return {"verified": False, "message": "Domain verification failed"}
            
    except Exception as e:
        logger.error(f"Error verifying DNS: {e}")
        return {"verified": False, "message": f"Verification error: {str(e)}"}

# FIXED: Changed from path parameter to POST request to handle complex URLs
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
            
            var botContainer = document.createElement('div');
            botContainer.id = 'nav-bot-container';
            botContainer.style.cssText = 'position: fixed; bottom: 20px; right: 20px; z-index: 9999; width: 350px; height: 500px; background: white; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.15); display: none;';
            
            var botToggle = document.createElement('button');
            botToggle.innerHTML = '💬';
            botToggle.style.cssText = 'position: fixed; bottom: 20px; right: 20px; z-index: 10000; width: 60px; height: 60px; border-radius: 50%; border: none; background: ' + botConfig.color + '; color: white; font-size: 24px; cursor: pointer; box-shadow: 0 4px 20px rgba(0,0,0,0.15);';
            
            botToggle.onclick = function() {{
                if (botContainer.style.display === 'none') {{
                    botContainer.style.display = 'block';
                    botToggle.innerHTML = '✕';
                }} else {{
                    botContainer.style.display = 'none';
                    botToggle.innerHTML = '💬';
                }}
            }};
            
            var iframe = document.createElement('iframe');
            iframe.src = botConfig.apiUrl + '/widget';
            iframe.style.cssText = 'width: 100%; height: 100%; border: none; border-radius: 10px;';
            
            botContainer.appendChild(iframe);
            document.body.appendChild(botContainer);
            document.body.appendChild(botToggle);
        }})();
    </script>
    """
    
    return {"widget_code": widget_code, "domain": domain}

# ALTERNATIVE: Keep the GET endpoint but with URL decoding
@app.get("/api/widget-code/{domain:path}")
async def get_widget_code_get(domain: str):
    """Generate widget code for integration (GET method with URL decoding)"""
    # Decode the URL-encoded domain
    decoded_domain = unquote(domain)
    
    # Get bot configuration if exists
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
    
    widget_code = f"""
    <!-- Navigation Helper Bot Widget -->
    <div id="nav-helper-bot"></div>
    <script>
        (function() {{
            var botConfig = {{
                domain: '{decoded_domain}',
                apiUrl: 'http://localhost:8000',
                position: '{widget_position}',
                color: '{widget_color}'
            }};
            
            var botContainer = document.createElement('div');
            botContainer.id = 'nav-bot-container';
            botContainer.style.cssText = 'position: fixed; bottom: 20px; right: 20px; z-index: 9999; width: 350px; height: 500px; background: white; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.15); display: none;';
            
            var botToggle = document.createElement('button');
            botToggle.innerHTML = '💬';
            botToggle.style.cssText = 'position: fixed; bottom: 20px; right: 20px; z-index: 10000; width: 60px; height: 60px; border-radius: 50%; border: none; background: ' + botConfig.color + '; color: white; font-size: 24px; cursor: pointer; box-shadow: 0 4px 20px rgba(0,0,0,0.15);';
            
            botToggle.onclick = function() {{
                if (botContainer.style.display === 'none') {{
                    botContainer.style.display = 'block';
                    botToggle.innerHTML = '✕';
                }} else {{
                    botContainer.style.display = 'none';
                    botToggle.innerHTML = '💬';
                }}
            }};
            
            var iframe = document.createElement('iframe');
            iframe.src = botConfig.apiUrl + '/widget';
            iframe.style.cssText = 'width: 100%; height: 100%; border: none; border-radius: 10px;';
            
            botContainer.appendChild(iframe);
            document.body.appendChild(botContainer);
            document.body.appendChild(botToggle);
        }})();
    </script>
    """
    
    return {"widget_code": widget_code, "domain": decoded_domain}

@app.get("/widget")
async def widget_interface():
    """Return the widget HTML interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Navigation Helper Bot</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            .chat-container { height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; }
            .message { margin: 10px 0; padding: 8px; border-radius: 5px; }
            .user-message { background: #007bff; color: white; text-align: right; }
            .bot-message { background: #f8f9fa; border: 1px solid #ddd; }
            .input-container { display: flex; }
            .input-container input { flex: 1; padding: 10px; border: 1px solid #ddd; }
            .input-container button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="chat-container" id="chatContainer">
            <div class="message bot-message">
                Hello! I'm your navigation helper. How can I assist you today?
            </div>
        </div>
        <div class="input-container">
            <input type="text" id="messageInput" placeholder="Type your message..." onkeypress="if(event.key==='Enter') sendMessage()">
            <button onclick="sendMessage()">Send</button>
        </div>
        
        <script>
            let sessionId = Math.random().toString(36).substring(7);
            
            function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (!message) return;
                
                const chatContainer = document.getElementById('chatContainer');
                
                // Add user message
                chatContainer.innerHTML += '<div class="message user-message">' + message + '</div>';
                
                // Send to API
                fetch('/api/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message: message, session_id: sessionId})
                })
                .then(response => response.json())
                .then(data => {
                    chatContainer.innerHTML += '<div class="message bot-message">' + data.response + '</div>';
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                })
                .catch(error => {
                    chatContainer.innerHTML += '<div class="message bot-message">Sorry, there was an error processing your message.</div>';
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                });
                
                input.value = '';
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/knowledge-base")
async def get_knowledge_base():
    """Get all documents in knowledge base"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename, file_type, upload_date FROM knowledge_base ORDER BY upload_date DESC")
        documents = [dict(row) for row in cursor.fetchall()]
    
    return {"documents": documents}

@app.get("/api/sitemap")
async def get_sitemap():
    """Get generated sitemap"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT url, title, depth FROM sitemap ORDER BY depth, url")
        sitemap = [dict(row) for row in cursor.fetchall()]
    
    return {"sitemap": sitemap}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)