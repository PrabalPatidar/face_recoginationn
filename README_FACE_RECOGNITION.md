# Face Recognition System with Supabase Integration

A Python web application for face recognition that uses Supabase for data storage and similarity search. Built with FastAPI and featuring a modern, responsive frontend.

## Features

- 🔍 **Face Detection & Recognition**: Upload images to detect and recognize faces
- 🗄️ **Supabase Integration**: Store face embeddings and person details in Supabase
- 📊 **Similarity Search**: Find similar faces using cosine similarity
- 👤 **Person Management**: Add new persons with their details
- 🎨 **Modern UI**: Beautiful, responsive web interface
- ⚡ **Fast Processing**: Efficient face encoding and recognition

## Technology Stack

- **Backend**: FastAPI (Python)
- **Database**: Supabase (PostgreSQL with vector extensions)
- **Face Recognition**: face_recognition library
- **Frontend**: HTML, CSS, JavaScript
- **Image Processing**: OpenCV, PIL

## Prerequisites

- Python 3.8+
- Supabase account and project
- Git

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd face-recognition-supabase
```

### 2. Set Up Supabase

1. Create a new project at [supabase.com](https://supabase.com)
2. Go to your project's SQL Editor
3. Run the SQL schema from `supabase_schema.sql`:

```sql
-- Copy and paste the contents of supabase_schema.sql
-- This will create the necessary tables and functions
```

4. Get your project URL and anon key from Settings > API

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

1. Copy the example environment file:
```bash
cp env.example .env
```

2. Edit `.env` with your Supabase credentials:
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
```

### 5. Run the Application

```bash
python app.py
```

The application will be available at `http://localhost:8000`

## Usage

### 1. Upload and Recognize Faces

1. Open your browser and go to `http://localhost:8000`
2. Click "Choose Image" or drag and drop an image
3. The system will:
   - Detect faces in the image
   - Generate face embeddings
   - Search for similar faces in the database
   - Display results or prompt to add a new person

### 2. Add New Persons

1. Upload an image with a face
2. If no match is found, click "Add New Person"
3. Fill in the person's details (name, email, phone)
4. Click "Add Person" to save to the database

### 3. API Endpoints

The application provides REST API endpoints:

- `POST /api/recognize` - Recognize a face from uploaded image
- `POST /api/add-person` - Add a new person with face embedding
- `GET /api/persons` - Get all persons in the database
- `GET /api/health` - Health check endpoint

## Database Schema

### Tables

#### `persons`
- `id` (UUID, Primary Key)
- `name` (VARCHAR, Required)
- `email` (VARCHAR, Optional)
- `phone` (VARCHAR, Optional)
- `created_at` (TIMESTAMP)
- `updated_at` (TIMESTAMP)

#### `face_embeddings`
- `id` (UUID, Primary Key)
- `person_id` (UUID, Foreign Key)
- `embedding` (VECTOR(128))
- `image_url` (TEXT, Optional)
- `confidence_score` (FLOAT, Optional)
- `created_at` (TIMESTAMP)
- `updated_at` (TIMESTAMP)

### Functions

- `find_similar_faces()` - Find faces similar to a query embedding
- `find_best_face_match()` - Find the best matching face
- `update_updated_at_column()` - Auto-update timestamps

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SUPABASE_URL` | Your Supabase project URL | Required |
| `SUPABASE_ANON_KEY` | Your Supabase anon key | Required |
| `SIMILARITY_THRESHOLD` | Minimum similarity for face matches | 0.6 |
| `MAX_FILE_SIZE` | Maximum upload file size (bytes) | 10485760 |

### Similarity Threshold

The similarity threshold determines how similar faces need to be to be considered a match:
- **0.6**: Default, good balance
- **0.7**: More strict, fewer false positives
- **0.5**: More lenient, more false positives

## Development

### Project Structure

```
face-recognition-supabase/
├── app.py                 # Main FastAPI application
├── requirements.txt       # Python dependencies
├── supabase_schema.sql    # Database schema
├── supabase_migration.sql # Migration script
├── templates/
│   └── index.html        # Frontend interface
├── static/               # Static files (if needed)
└── README.md            # This file
```

### Running in Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Testing

Test the API endpoints using curl or a tool like Postman:

```bash
# Health check
curl http://localhost:8000/api/health

# Recognize face
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/api/recognize

# Add person
curl -X POST -F "file=@test_image.jpg" -F "name=John Doe" -F "email=john@example.com" http://localhost:8000/api/add-person
```

## Troubleshooting

### Common Issues

1. **"No face detected"**
   - Ensure the image contains a clear, front-facing face
   - Try with a higher resolution image
   - Check that the face is well-lit

2. **Supabase connection errors**
   - Verify your SUPABASE_URL and SUPABASE_ANON_KEY
   - Check that the database schema has been applied
   - Ensure the vector extension is enabled

3. **Import errors**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

4. **File upload errors**
   - Check file size (max 10MB)
   - Ensure file is a valid image format
   - Verify file permissions

### Logs

Check the console output for detailed error messages. The application logs important events and errors.

## Security Considerations

- **API Keys**: Never commit your Supabase keys to version control
- **File Uploads**: Validate file types and sizes
- **Rate Limiting**: Consider implementing rate limiting for production
- **Authentication**: Add authentication for production use
- **CORS**: Configure CORS properly for production

## Performance Optimization

- **Vector Indexing**: The schema includes vector indexes for fast similarity search
- **Image Compression**: Consider compressing uploaded images
- **Caching**: Implement caching for frequently accessed data
- **Batch Processing**: For bulk operations, consider batch processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the Supabase documentation
3. Open an issue in the repository

## Acknowledgments

- [face_recognition](https://github.com/ageitgey/face_recognition) library
- [Supabase](https://supabase.com) for the backend infrastructure
- [FastAPI](https://fastapi.tiangolo.com) for the web framework
