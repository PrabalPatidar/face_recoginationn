# API Documentation

## Overview

The Face Scan Project provides a RESTful API for face detection and recognition services. This document describes all available endpoints, request/response formats, and authentication requirements.

## Base URL

```
http://localhost:5000/api/v1
```

## Authentication

All API endpoints require authentication using an API key. Include the API key in the request header:

```
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### Health Check

#### GET /health

Check the health status of the API service.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2023-12-01T10:00:00Z",
  "version": "1.0.0"
}
```

### Face Detection

#### POST /scan/detect

Detect faces in an uploaded image.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: image file

**Response:**
```json
{
  "success": true,
  "faces_detected": 2,
  "faces": [
    {
      "id": 1,
      "bounding_box": {
        "x": 100,
        "y": 150,
        "width": 80,
        "height": 80
      },
      "confidence": 0.95
    },
    {
      "id": 2,
      "bounding_box": {
        "x": 300,
        "y": 200,
        "width": 75,
        "height": 75
      },
      "confidence": 0.87
    }
  ],
  "processing_time": 0.234
}
```

### Face Recognition

#### POST /scan/recognize

Recognize faces in an uploaded image against known faces.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: image file

**Response:**
```json
{
  "success": true,
  "faces_recognized": 2,
  "faces": [
    {
      "id": 1,
      "bounding_box": {
        "x": 100,
        "y": 150,
        "width": 80,
        "height": 80
      },
      "name": "John Doe",
      "confidence": 0.92,
      "encoding_id": "enc_123"
    },
    {
      "id": 2,
      "bounding_box": {
        "x": 300,
        "y": 200,
        "width": 75,
        "height": 75
      },
      "name": "Unknown",
      "confidence": 0.0,
      "encoding_id": null
    }
  ],
  "processing_time": 0.456
}
```

### Face Management

#### GET /faces

Retrieve all known faces.

**Response:**
```json
{
  "success": true,
  "faces": [
    {
      "id": 1,
      "name": "John Doe",
      "encoding_id": "enc_123",
      "created_at": "2023-12-01T10:00:00Z",
      "updated_at": "2023-12-01T10:00:00Z"
    },
    {
      "id": 2,
      "name": "Jane Smith",
      "encoding_id": "enc_456",
      "created_at": "2023-12-01T11:00:00Z",
      "updated_at": "2023-12-01T11:00:00Z"
    }
  ],
  "total_count": 2
}
```

#### POST /faces

Add a new face to the recognition database.

**Request:**
```json
{
  "name": "New Person",
  "image": "base64_encoded_image_data"
}
```

**Response:**
```json
{
  "success": true,
  "face": {
    "id": 3,
    "name": "New Person",
    "encoding_id": "enc_789",
    "created_at": "2023-12-01T12:00:00Z"
  }
}
```

#### GET /faces/{face_id}

Retrieve a specific face by ID.

**Response:**
```json
{
  "success": true,
  "face": {
    "id": 1,
    "name": "John Doe",
    "encoding_id": "enc_123",
    "created_at": "2023-12-01T10:00:00Z",
    "updated_at": "2023-12-01T10:00:00Z"
  }
}
```

#### PUT /faces/{face_id}

Update a face's information.

**Request:**
```json
{
  "name": "Updated Name"
}
```

**Response:**
```json
{
  "success": true,
  "face": {
    "id": 1,
    "name": "Updated Name",
    "encoding_id": "enc_123",
    "created_at": "2023-12-01T10:00:00Z",
    "updated_at": "2023-12-01T12:00:00Z"
  }
}
```

#### DELETE /faces/{face_id}

Delete a face from the recognition database.

**Response:**
```json
{
  "success": true,
  "message": "Face deleted successfully"
}
```

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": "Additional error details"
  }
}
```

### Common Error Codes

- `INVALID_REQUEST`: Invalid request format or missing required fields
- `UNAUTHORIZED`: Missing or invalid API key
- `FORBIDDEN`: Insufficient permissions
- `NOT_FOUND`: Resource not found
- `VALIDATION_ERROR`: Input validation failed
- `PROCESSING_ERROR`: Error during face processing
- `STORAGE_ERROR`: Error accessing storage
- `INTERNAL_ERROR`: Internal server error

## Rate Limiting

API requests are rate limited to prevent abuse:

- 100 requests per minute per API key
- 1000 requests per hour per API key

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## WebSocket Support

### Real-time Face Detection

Connect to the WebSocket endpoint for real-time face detection:

```
ws://localhost:5000/ws/detect
```

**Message Format:**
```json
{
  "type": "frame",
  "data": "base64_encoded_image_data"
}
```

**Response:**
```json
{
  "type": "detection_result",
  "faces": [
    {
      "bounding_box": {
        "x": 100,
        "y": 150,
      "width": 80,
        "height": 80
      },
      "confidence": 0.95
    }
  ],
  "timestamp": "2023-12-01T10:00:00Z"
}
```

## SDK Examples

### Python

```python
import requests

# Face detection
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/api/v1/scan/detect',
        files={'image': f},
        headers={'Authorization': 'Bearer YOUR_API_KEY'}
    )
    result = response.json()
```

### JavaScript

```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('http://localhost:5000/api/v1/scan/detect', {
    method: 'POST',
    headers: {
        'Authorization': 'Bearer YOUR_API_KEY'
    },
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## Testing

Use the provided test endpoints for development:

- `GET /test/health` - Test health check
- `POST /test/detect` - Test face detection with sample image
- `GET /test/faces` - Test face retrieval

## Support

For API support and questions:
- Email: support@facescan.com
- Documentation: https://docs.facescan.com
- GitHub Issues: https://github.com/yourusername/face-scan-project/issues
