"""
Simple FastAPI Face Recognition Application (without face recognition for now)
"""

import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="Face recognition system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Pydantic models
class PersonCreate(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None

class PersonResponse(BaseModel):
    id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class FaceMatchResponse(BaseModel):
    person: Optional[PersonResponse] = None
    similarity_score: Optional[float] = None
    is_match: bool
    message: str

# Utility functions
def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    try:
        with open("templates/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Face Recognition System</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
                .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
                .btn:hover { background: #0056b3; }
                .result { margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 Face Recognition System</h1>
                <p>Upload an image to recognize faces or add a new person to the database.</p>
                
                <div class="upload-area">
                    <h3>Upload Image for Recognition</h3>
                    <input type="file" id="imageInput" accept="image/*">
                    <br><br>
                    <button class="btn" onclick="recognizeFace()">Recognize Face</button>
                </div>
                
                <div class="upload-area">
                    <h3>Add New Person</h3>
                    <input type="file" id="newPersonImage" accept="image/*">
                    <br><br>
                    <input type="text" id="personName" placeholder="Person Name" style="padding: 10px; margin: 5px;">
                    <input type="email" id="personEmail" placeholder="Email (optional)" style="padding: 10px; margin: 5px;">
                    <br><br>
                    <button class="btn" onclick="addPerson()">Add Person</button>
                </div>
                
                <div id="result" class="result" style="display: none;">
                    <h3>Result</h3>
                    <p id="resultText"></p>
                </div>
            </div>
            
            <script>
                async function recognizeFace() {
                    const fileInput = document.getElementById('imageInput');
                    const file = fileInput.files[0];
                    
                    if (!file) {
                        alert('Please select an image file');
                        return;
                    }
                    
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    try {
                        const response = await fetch('/api/recognize', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const result = await response.json();
                        showResult(result);
                    } catch (error) {
                        showResult({message: 'Error: ' + error.message, is_match: false});
                    }
                }
                
                async function addPerson() {
                    const fileInput = document.getElementById('newPersonImage');
                    const nameInput = document.getElementById('personName');
                    const emailInput = document.getElementById('personEmail');
                    
                    const file = fileInput.files[0];
                    const name = nameInput.value;
                    
                    if (!file || !name) {
                        alert('Please select an image file and enter a name');
                        return;
                    }
                    
                    const formData = new FormData();
                    formData.append('file', file);
                    formData.append('name', name);
                    formData.append('email', emailInput.value);
                    
                    try {
                        const response = await fetch('/api/add-person', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const result = await response.json();
                        showResult({message: 'Person added successfully! ID: ' + result.id, is_match: true});
                        
                        // Clear form
                        nameInput.value = '';
                        emailInput.value = '';
                        fileInput.value = '';
                    } catch (error) {
                        showResult({message: 'Error: ' + error.message, is_match: false});
                    }
                }
                
                function showResult(result) {
                    const resultDiv = document.getElementById('result');
                    const resultText = document.getElementById('resultText');
                    
                    resultDiv.style.display = 'block';
                    resultText.textContent = result.message;
                    
                    if (result.is_match) {
                        resultDiv.style.background = '#d4edda';
                        resultDiv.style.color = '#155724';
                    } else {
                        resultDiv.style.background = '#f8d7da';
                        resultDiv.style.color = '#721c24';
                    }
                }
            </script>
        </body>
        </html>
        """)

@app.post("/api/recognize", response_model=FaceMatchResponse)
async def recognize_face(file: UploadFile = File(...)):
    """Recognize a face from uploaded image."""
    try:
        # Validate file
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # For now, just return a mock response since face recognition is not available
        return FaceMatchResponse(
            is_match=False,
            message="Face recognition feature is temporarily disabled. Please install dlib and face-recognition libraries."
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in recognize_face: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/add-person")
async def add_person(
    file: UploadFile = File(...),
    name: str = Form(...),
    email: Optional[str] = Form(None),
    phone: Optional[str] = Form(None)
):
    """Add a new person with face embedding."""
    try:
        # Validate file
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # For now, just return a mock response
        return {
            "id": "mock-id-" + str(int(datetime.now().timestamp())),
            "person_id": "mock-person-id",
            "created_at": datetime.now()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in add_person: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now()}

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "app_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
