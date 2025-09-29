"""
Face Recognition Application with Basic Face Detection and Matching
"""

import os
import json
import uuid
import logging
import cv2
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="Face recognition system with OpenCV",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
DATA_DIR = Path("data")
PERSONS_FILE = DATA_DIR / "persons.json"
IMAGES_DIR = DATA_DIR / "images"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

# Load OpenCV face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
    image_path: Optional[str] = None
    face_features: Optional[List[float]] = None
    created_at: str
    updated_at: str

class FaceMatchResponse(BaseModel):
    person: Optional[PersonResponse] = None
    similarity_score: Optional[float] = None
    is_match: bool
    message: str

# Data storage functions
def load_persons() -> List[Dict]:
    """Load persons from JSON file."""
    if PERSONS_FILE.exists():
        try:
            with open(PERSONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading persons: {e}")
            return []
    return []

def save_persons(persons: List[Dict]) -> bool:
    """Save persons to JSON file."""
    try:
        with open(PERSONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(persons, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving persons: {e}")
        return False

def save_image(file: UploadFile) -> Optional[str]:
    """Save uploaded image and return the path."""
    try:
        # Generate unique filename
        file_extension = file.filename.split('.')[-1].lower()
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        image_path = IMAGES_DIR / unique_filename
        
        # Save file
        with open(image_path, 'wb') as f:
            content = file.file.read()
            f.write(content)
        
        return str(image_path.relative_to(DATA_DIR))
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        return None

def extract_face_features(image_path: str) -> Optional[List[float]]:
    """Extract basic face features using OpenCV."""
    try:
        # Read image
        image = cv2.imread(str(DATA_DIR / image_path))
        if image is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
        
        # Get the largest face
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        x, y, w, h = largest_face
        
        # Extract face region
        face_region = gray[y:y+h, x:x+w]
        
        # Resize to standard size for comparison
        face_region = cv2.resize(face_region, (100, 100))
        
        # Calculate basic features (histogram)
        features = []
        for i in range(10):  # 10 histogram bins
            hist = cv2.calcHist([face_region], [0], None, [25], [0, 256])
            features.extend(hist.flatten().tolist())
        
        # Add geometric features
        features.extend([
            w / h,  # aspect ratio
            x / image.shape[1],  # relative x position
            y / image.shape[0],  # relative y position
            w / image.shape[1],  # relative width
            h / image.shape[0]   # relative height
        ])
        
        return features
    
    except Exception as e:
        logger.error(f"Error extracting face features: {e}")
        return None

def compare_faces(features1: List[float], features2: List[float]) -> float:
    """Compare two face feature vectors and return similarity score."""
    try:
        if len(features1) != len(features2):
            return 0.0
        
        # Calculate cosine similarity
        dot_product = sum(a * b for a, b in zip(features1, features2))
        magnitude1 = sum(a * a for a in features1) ** 0.5
        magnitude2 = sum(a * a for a in features2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        similarity = dot_product / (magnitude1 * magnitude2)
        return max(0.0, similarity)  # Ensure non-negative
    
    except Exception as e:
        logger.error(f"Error comparing faces: {e}")
        return 0.0

def find_person_by_name(name: str) -> Optional[Dict]:
    """Find person by name (case-insensitive)."""
    persons = load_persons()
    for person in persons:
        if person['name'].lower() == name.lower():
            return person
    return None

def find_matching_face(uploaded_features: List[float], threshold: float = 0.7) -> Optional[Dict]:
    """Find matching face in database."""
    persons = load_persons()
    best_match = None
    best_score = 0.0
    
    for person in persons:
        if person.get('face_features'):
            similarity = compare_faces(uploaded_features, person['face_features'])
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = person
    
    if best_match:
        best_match['similarity_score'] = best_score
    
    return best_match

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

# API Routes
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Face Recognition System</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                max-width: 900px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 15px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .header h1 {
                margin: 0;
                font-size: 2.5em;
            }
            .content {
                padding: 30px;
            }
            .upload-area { 
                border: 2px dashed #ddd; 
                padding: 40px; 
                text-align: center; 
                margin: 20px 0; 
                border-radius: 10px;
                background: #f9f9f9;
                transition: all 0.3s ease;
            }
            .upload-area:hover {
                border-color: #667eea;
                background: #f0f4ff;
            }
            .btn { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; 
                padding: 12px 25px; 
                border: none; 
                border-radius: 25px; 
                cursor: pointer; 
                font-size: 16px;
                transition: transform 0.2s ease;
            }
            .btn:hover { 
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            .result { 
                margin: 20px 0; 
                padding: 20px; 
                border-radius: 10px;
                border-left: 5px solid;
            }
            .result.success {
                background: #d4edda;
                color: #155724;
                border-left-color: #28a745;
            }
            .result.error {
                background: #f8d7da;
                color: #721c24;
                border-left-color: #dc3545;
            }
            .result.info {
                background: #d1ecf1;
                color: #0c5460;
                border-left-color: #17a2b8;
            }
            .person-card {
                background: white;
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 20px;
                margin: 15px 0;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .person-image {
                width: 80px;
                height: 80px;
                border-radius: 50%;
                object-fit: cover;
                float: left;
                margin-right: 15px;
            }
            .person-info h3 {
                margin: 0 0 10px 0;
                color: #333;
            }
            .person-info p {
                margin: 5px 0;
                color: #666;
            }
            .input-group {
                margin: 10px 0;
            }
            .input-group input {
                width: 100%;
                padding: 12px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 16px;
                box-sizing: border-box;
            }
            .stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .stat-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            }
            .stat-number {
                font-size: 2em;
                font-weight: bold;
                margin-bottom: 5px;
            }
            #imagePreview {
                max-width: 200px;
                max-height: 200px;
                border-radius: 10px;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Face Recognition System</h1>
                <p>Upload an image to recognize faces or add a new person to the database</p>
            </div>
            
            <div class="content">
                <div class="stats" id="stats">
                    <div class="stat-card">
                        <div class="stat-number" id="totalPersons">0</div>
                        <div>Total Persons</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number" id="totalRecognitions">0</div>
                        <div>Recognitions Today</div>
                    </div>
                </div>
                
                <div class="upload-area">
                    <h3>📷 Upload Image for Recognition</h3>
                    <input type="file" id="imageInput" accept="image/*" onchange="previewImage(this)">
                    <br><br>
                    <img id="imagePreview" style="display: none;">
                    <br>
                    <button class="btn" onclick="recognizeFace()">🔍 Recognize Face</button>
                </div>
                
                <div class="upload-area">
                    <h3>👤 Add New Person</h3>
                    <input type="file" id="newPersonImage" accept="image/*" onchange="previewNewPersonImage(this)">
                    <br><br>
                    <img id="newPersonPreview" style="display: none;">
                    <br>
                    <div class="input-group">
                        <input type="text" id="personName" placeholder="Person Name *" required>
                    </div>
                    <div class="input-group">
                        <input type="email" id="personEmail" placeholder="Email (optional)">
                    </div>
                    <div class="input-group">
                        <input type="tel" id="personPhone" placeholder="Phone (optional)">
                    </div>
                    <button class="btn" onclick="addPerson()">➕ Add Person</button>
                </div>
                
                <div id="result" class="result" style="display: none;">
                    <h3>Result</h3>
                    <div id="resultContent"></div>
                </div>
                
                <div id="personsList">
                    <h3>📋 Registered Persons</h3>
                    <div id="personsContainer"></div>
                </div>
            </div>
        </div>
        
        <script>
            let recognitionCount = 0;
            
            // Load initial data
            loadPersons();
            updateStats();
            
            function previewImage(input) {
                const preview = document.getElementById('imagePreview');
                if (input.files && input.files[0]) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                    }
                    reader.readAsDataURL(input.files[0]);
                }
            }
            
            function previewNewPersonImage(input) {
                const preview = document.getElementById('newPersonPreview');
                if (input.files && input.files[0]) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        preview.src = e.target.result;
                        preview.style.display = 'block';
                    }
                    reader.readAsDataURL(input.files[0]);
                }
            }
            
            async function recognizeFace() {
                const fileInput = document.getElementById('imageInput');
                const file = fileInput.files[0];
                
                if (!file) {
                    showResult('Please select an image file', 'error');
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
                    showRecognitionResult(result);
                    recognitionCount++;
                    updateStats();
                } catch (error) {
                    showResult('Error: ' + error.message, 'error');
                }
            }
            
            async function addPerson() {
                const fileInput = document.getElementById('newPersonImage');
                const nameInput = document.getElementById('personName');
                const emailInput = document.getElementById('personEmail');
                const phoneInput = document.getElementById('personPhone');
                
                const file = fileInput.files[0];
                const name = nameInput.value.trim();
                
                if (!file || !name) {
                    showResult('Please select an image file and enter a name', 'error');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                formData.append('name', name);
                formData.append('email', emailInput.value.trim());
                formData.append('phone', phoneInput.value.trim());
                
                try {
                    const response = await fetch('/api/add-person', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    showResult(`Person added successfully! ID: ${result.id}`, 'success');
                    
                    // Clear form
                    nameInput.value = '';
                    emailInput.value = '';
                    phoneInput.value = '';
                    fileInput.value = '';
                    document.getElementById('newPersonPreview').style.display = 'none';
                    
                    // Reload persons list
                    loadPersons();
                    updateStats();
                } catch (error) {
                    showResult('Error: ' + error.message, 'error');
                }
            }
            
            async function loadPersons() {
                try {
                    const response = await fetch('/api/persons');
                    const persons = await response.json();
                    displayPersons(persons);
                } catch (error) {
                    console.error('Error loading persons:', error);
                }
            }
            
            function displayPersons(persons) {
                const container = document.getElementById('personsContainer');
                if (persons.length === 0) {
                    container.innerHTML = '<p>No persons registered yet.</p>';
                    return;
                }
                
                container.innerHTML = persons.map(person => `
                    <div class="person-card">
                        <img src="/api/image/${person.image_path}" class="person-image" 
                             alt="${person.name}" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODAiIGhlaWdodD0iODAiIHZpZXdCb3g9IjAgMCA4MCA4MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGNpcmNsZSBjeD0iNDAiIGN5PSI0MCIgcj0iNDAiIGZpbGw9IiNFNUU3RUIiLz4KPHN2ZyB4PSIyMCIgeT0iMjAiIHdpZHRoPSI0MCIgaGVpZ2h0PSI0MCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIj4KPHBhdGggZD0iTTEyIDEyQzE0LjIwOTEgMTIgMTYgMTAuMjA5MSAxNiA4QzE2IDUuNzkwODYgMTQuMjA5MSA0IDEyIDRDOS43OTA4NiA0IDggNS43OTA4NiA4IDhDOCAxMC4yMDkxIDkuNzkwODYgMTIgMTIgMTJaIiBmaWxsPSIjOUM5Q0E1Ii8+CjxwYXRoIGQ9Ik0xMiAxNEM5LjMzIDE0IDcgMTYuMzMgNyAxOVYyMEgxN1YxOUMxNyAxNi4zMyAxNC42NyAxNCAxMiAxNFoiIGZpbGw9IiM5QzlDQTUiLz4KPC9zdmc+Cjwvc3ZnPgo='">
                        <div class="person-info">
                            <h3>${person.name}</h3>
                            <p><strong>Email:</strong> ${person.email || 'Not provided'}</p>
                            <p><strong>Phone:</strong> ${person.phone || 'Not provided'}</p>
                            <p><strong>Added:</strong> ${new Date(person.created_at).toLocaleDateString()}</p>
                        </div>
                        <div style="clear: both;"></div>
                    </div>
                `).join('');
            }
            
            function showResult(message, type) {
                const resultDiv = document.getElementById('result');
                const resultContent = document.getElementById('resultContent');
                
                resultDiv.style.display = 'block';
                resultDiv.className = `result ${type}`;
                resultContent.innerHTML = `<p>${message}</p>`;
                
                // Auto-hide after 5 seconds
                setTimeout(() => {
                    resultDiv.style.display = 'none';
                }, 5000);
            }
            
            function showRecognitionResult(result) {
                const resultDiv = document.getElementById('result');
                const resultContent = document.getElementById('resultContent');
                
                resultDiv.style.display = 'block';
                
                if (result.is_match && result.person) {
                    resultDiv.className = 'result success';
                    resultContent.innerHTML = `
                        <p><strong>✅ Face Recognized!</strong></p>
                        <p><strong>Name:</strong> ${result.person.name}</p>
                        <p><strong>Email:</strong> ${result.person.email || 'Not provided'}</p>
                        <p><strong>Phone:</strong> ${result.person.phone || 'Not provided'}</p>
                        <p><strong>Similarity:</strong> ${(result.similarity_score * 100).toFixed(1)}%</p>
                    `;
                } else {
                    resultDiv.className = 'result info';
                    resultContent.innerHTML = `
                        <p><strong>❌ Face Not Found</strong></p>
                        <p>${result.message}</p>
                        <p>Would you like to add this person to the database?</p>
                    `;
                }
            }
            
            function updateStats() {
                document.getElementById('totalRecognitions').textContent = recognitionCount;
                
                // Update total persons count
                fetch('/api/persons')
                    .then(response => response.json())
                    .then(persons => {
                        document.getElementById('totalPersons').textContent = persons.length;
                    })
                    .catch(error => console.error('Error updating stats:', error));
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
        
        # Process image
        image = process_uploaded_image(file)
        
        # Save temporarily to extract features
        temp_filename = f"temp_{uuid.uuid4()}.jpg"
        temp_path = IMAGES_DIR / temp_filename
        cv2.imwrite(str(temp_path), image)
        
        try:
            # Extract face features
            features = extract_face_features(str(temp_path.relative_to(DATA_DIR)))
            
            if features is None:
                return FaceMatchResponse(
                    is_match=False,
                    message="No face detected in the image"
                )
            
            # Find matching face
            match_result = find_matching_face(features, threshold=0.6)
            
            if match_result:
                person = PersonResponse(**match_result)
                return FaceMatchResponse(
                    person=person,
                    similarity_score=match_result['similarity_score'],
                    is_match=True,
                    message=f"Face recognized as {match_result['name']}"
                )
            else:
                return FaceMatchResponse(
                    is_match=False,
                    message="No matching face found in database"
                )
        
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
    
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
    """Add a new person with image."""
    try:
        # Validate file
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # Check if person already exists
        existing_person = find_person_by_name(name)
        if existing_person:
            raise HTTPException(status_code=400, detail="Person with this name already exists")
        
        # Save image
        image_path = save_image(file)
        if not image_path:
            raise HTTPException(status_code=500, detail="Failed to save image")
        
        # Extract face features
        face_features = extract_face_features(image_path)
        
        # Create person data
        person_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        person_data = {
            "id": person_id,
            "name": name,
            "email": email,
            "phone": phone,
            "image_path": image_path,
            "face_features": face_features,
            "created_at": now,
            "updated_at": now
        }
        
        # Load existing persons and add new one
        persons = load_persons()
        persons.append(person_data)
        
        # Save to file
        if not save_persons(persons):
            raise HTTPException(status_code=500, detail="Failed to save person data")
        
        return {
            "id": person_id,
            "message": "Person added successfully with face recognition features"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in add_person: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/persons", response_model=List[PersonResponse])
async def get_persons():
    """Get all persons in the database."""
    try:
        persons = load_persons()
        return [PersonResponse(**person) for person in persons]
    except Exception as e:
        logger.error(f"Error getting persons: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/image/{image_path:path}")
async def get_image(image_path: str):
    """Serve stored images."""
    try:
        full_path = DATA_DIR / image_path
        if full_path.exists() and full_path.is_file():
            return FileResponse(full_path)
        else:
            raise HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        logger.error(f"Error serving image: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

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
        "app_with_recognition:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

