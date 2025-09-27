"""
FastAPI Face Recognition Application with Supabase Integration
"""

import os
import logging
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime
import cv2
import face_recognition
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
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
    description="Face recognition system with Supabase integration",
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

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment variables")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configuration
SIMILARITY_THRESHOLD = 0.6
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

class FaceEmbeddingResponse(BaseModel):
    id: str
    person_id: str
    confidence_score: Optional[float] = None
    created_at: datetime

# Utility functions
def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_uploaded_image(file: UploadFile) -> np.ndarray:
    """Process uploaded image file and return as numpy array."""
    try:
        # Read file content
        file_content = file.file.read()
        
        # Check file size
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        # Convert to numpy array
        nparr = np.frombuffer(file_content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        return image
    except Exception as e:
        logger.error(f"Error processing uploaded image: {e}")
        raise HTTPException(status_code=400, detail="Error processing image")

def detect_and_encode_face(image: np.ndarray) -> tuple:
    """Detect face in image and return encoding."""
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Find face locations
        face_locations = face_recognition.face_locations(rgb_image)
        
        if not face_locations:
            return None, None
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if not face_encodings:
            return None, None
        
        # Return first face encoding and location
        return face_encodings[0], face_locations[0]
    
    except Exception as e:
        logger.error(f"Error detecting face: {e}")
        return None, None

def find_similar_face(embedding: np.ndarray) -> Optional[Dict[str, Any]]:
    """Find similar face in Supabase database."""
    try:
        # Convert numpy array to list for JSON serialization
        embedding_list = embedding.tolist()
        
        # Call the similarity search function
        result = supabase.rpc(
            'find_best_face_match',
            {
                'query_embedding': embedding_list,
                'similarity_threshold': SIMILARITY_THRESHOLD
            }
        ).execute()
        
        if result.data and len(result.data) > 0:
            return result.data[0]
        
        return None
    
    except Exception as e:
        logger.error(f"Error finding similar face: {e}")
        return None

def save_face_embedding(person_id: str, embedding: np.ndarray, confidence_score: float = None) -> str:
    """Save face embedding to Supabase."""
    try:
        # Convert numpy array to list
        embedding_list = embedding.tolist()
        
        # Insert into face_embeddings table
        result = supabase.table('face_embeddings').insert({
            'person_id': person_id,
            'embedding': embedding_list,
            'confidence_score': confidence_score
        }).execute()
        
        if result.data and len(result.data) > 0:
            return result.data[0]['id']
        
        raise Exception("Failed to save embedding")
    
    except Exception as e:
        logger.error(f"Error saving face embedding: {e}")
        raise HTTPException(status_code=500, detail="Error saving face embedding")

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    try:
        with open("templates/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Face Recognition System</h1><p>Frontend not found</p>")

@app.post("/api/recognize", response_model=FaceMatchResponse)
async def recognize_face(file: UploadFile = File(...)):
    """Recognize a face from uploaded image."""
    try:
        # Validate file
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # Process image
        image = process_uploaded_image(file)
        
        # Detect and encode face
        face_encoding, face_location = detect_and_encode_face(image)
        
        if face_encoding is None:
            return FaceMatchResponse(
                is_match=False,
                message="No face detected in the image"
            )
        
        # Find similar face
        match_result = find_similar_face(face_encoding)
        
        if match_result:
            # Create person response
            person = PersonResponse(
                id=match_result['person_id'],
                name=match_result['person_name'],
                email=match_result['person_email'],
                phone=match_result['person_phone'],
                created_at=datetime.fromisoformat(match_result['created_at'].replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(match_result['created_at'].replace('Z', '+00:00'))
            )
            
            return FaceMatchResponse(
                person=person,
                similarity_score=match_result['similarity_score'],
                is_match=True,
                message=f"Face recognized as {match_result['person_name']}"
            )
        else:
            return FaceMatchResponse(
                is_match=False,
                message="No matching face found in database"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in recognize_face: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/add-person", response_model=FaceEmbeddingResponse)
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
        
        # Process image
        image = process_uploaded_image(file)
        
        # Detect and encode face
        face_encoding, face_location = detect_and_encode_face(image)
        
        if face_encoding is None:
            raise HTTPException(status_code=400, detail="No face detected in the image")
        
        # Create person
        person_data = {
            'name': name,
            'email': email,
            'phone': phone
        }
        
        # Insert person into database
        person_result = supabase.table('persons').insert(person_data).execute()
        
        if not person_result.data or len(person_result.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to create person")
        
        person_id = person_result.data[0]['id']
        
        # Save face embedding
        embedding_id = save_face_embedding(person_id, face_encoding)
        
        return FaceEmbeddingResponse(
            id=embedding_id,
            person_id=person_id,
            created_at=datetime.now()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in add_person: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/persons", response_model=List[PersonResponse])
async def get_persons():
    """Get all persons in the database."""
    try:
        result = supabase.table('persons').select('*').execute()
        
        persons = []
        for person_data in result.data:
            person = PersonResponse(
                id=person_data['id'],
                name=person_data['name'],
                email=person_data['email'],
                phone=person_data['phone'],
                created_at=datetime.fromisoformat(person_data['created_at'].replace('Z', '+00:00')),
                updated_at=datetime.fromisoformat(person_data['updated_at'].replace('Z', '+00:00'))
            )
            persons.append(person)
        
        return persons
    
    except Exception as e:
        logger.error(f"Error getting persons: {e}")
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
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
