# Face Scan Project API Documentation

This directory contains comprehensive API documentation for the Face Scan Project.

## API Overview

The Face Scan Project provides a RESTful API for face detection and recognition services. The API is built using Flask and provides endpoints for:

- Face detection in images
- Face recognition and matching
- User management and authentication
- System monitoring and health checks
- Model management

## API Base URL

```
http://localhost:5000/api/v1
```

## Authentication

The API uses JWT (JSON Web Token) authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## Rate Limiting

API endpoints are rate-limited based on user tier:
- **Free**: 10 requests/minute
- **Premium**: 100 requests/minute
- **Enterprise**: 1000 requests/minute
- **Admin**: 10000 requests/minute

## Response Format

All API responses follow a consistent format:

### Success Response
```json
{
  "status": "success",
  "data": { ... },
  "message": "Operation completed successfully"
}
```

### Error Response
```json
{
  "status": "error",
  "error": "Error message",
  "code": "ERROR_CODE",
  "details": { ... }
}
```

## API Endpoints

### Authentication Endpoints
- [POST /auth/login](auth.md#login) - User login
- [POST /auth/logout](auth.md#logout) - User logout
- [POST /auth/refresh](auth.md#refresh) - Refresh access token
- [GET /auth/me](auth.md#me) - Get current user info

### Face Detection Endpoints
- [POST /faces/detect](faces.md#detect) - Detect faces in image
- [GET /faces/detect/{id}](faces.md#get-detect-result) - Get detection result
- [GET /faces/detect](faces.md#list-detect-results) - List detection results

### Face Recognition Endpoints
- [POST /faces/recognize](faces.md#recognize) - Recognize faces in image
- [POST /faces/compare](faces.md#compare) - Compare two face images
- [GET /faces/recognize/{id}](faces.md#get-recognition-result) - Get recognition result
- [GET /faces/recognize](faces.md#list-recognition-results) - List recognition results

### Face Management Endpoints
- [POST /faces/register](faces.md#register) - Register a new face
- [GET /faces/registered](faces.md#list-registered) - List registered faces
- [GET /faces/registered/{id}](faces.md#get-registered) - Get registered face
- [PUT /faces/registered/{id}](faces.md#update-registered) - Update registered face
- [DELETE /faces/registered/{id}](faces.md#delete-registered) - Delete registered face

### Scan Endpoints
- [POST /scan/image](scan.md#scan-image) - Scan image for faces
- [POST /scan/video](scan.md#scan-video) - Scan video for faces
- [GET /scan/{id}](scan.md#get-scan-result) - Get scan result
- [GET /scan](scan.md#list-scans) - List scan results

### Health and Monitoring Endpoints
- [GET /health](health.md#health-check) - System health check
- [GET /health/detailed](health.md#detailed-health) - Detailed health status
- [GET /metrics](health.md#metrics) - System metrics
- [GET /alerts](health.md#alerts) - Active alerts

### User Management Endpoints
- [GET /users](users.md#list-users) - List users (Admin only)
- [POST /users](users.md#create-user) - Create user (Admin only)
- [GET /users/{id}](users.md#get-user) - Get user details
- [PUT /users/{id}](users.md#update-user) - Update user
- [DELETE /users/{id}](users.md#delete-user) - Delete user (Admin only)

### Model Management Endpoints
- [GET /models](models.md#list-models) - List available models
- [POST /models/train](models.md#train-model) - Train new model (Admin only)
- [GET /models/{id}](models.md#get-model) - Get model details
- [POST /models/{id}/deploy](models.md#deploy-model) - Deploy model (Admin only)
- [DELETE /models/{id}](models.md#delete-model) - Delete model (Admin only)

## Error Codes

| Code | Description |
|------|-------------|
| `AUTH_REQUIRED` | Authentication required |
| `AUTH_INVALID` | Invalid authentication token |
| `AUTH_EXPIRED` | Authentication token expired |
| `PERMISSION_DENIED` | Insufficient permissions |
| `RATE_LIMIT_EXCEEDED` | Rate limit exceeded |
| `INVALID_INPUT` | Invalid input data |
| `FILE_TOO_LARGE` | File size exceeds limit |
| `UNSUPPORTED_FORMAT` | Unsupported file format |
| `MODEL_NOT_FOUND` | Model not found |
| `FACE_NOT_DETECTED` | No faces detected in image |
| `FACE_NOT_RECOGNIZED` | Face not recognized |
| `INTERNAL_ERROR` | Internal server error |

## SDKs and Libraries

### Python SDK
```python
from face_scan_sdk import FaceScanClient

client = FaceScanClient(api_key="your-api-key", base_url="http://localhost:5000/api/v1")

# Detect faces
result = client.detect_faces("path/to/image.jpg")

# Recognize faces
result = client.recognize_faces("path/to/image.jpg")
```

### JavaScript SDK
```javascript
import { FaceScanClient } from 'face-scan-sdk';

const client = new FaceScanClient({
  apiKey: 'your-api-key',
  baseUrl: 'http://localhost:5000/api/v1'
});

// Detect faces
const result = await client.detectFaces('path/to/image.jpg');

// Recognize faces
const result = await client.recognizeFaces('path/to/image.jpg');
```

## Webhooks

The API supports webhooks for real-time notifications:

### Webhook Events
- `face.detected` - Face detected in image
- `face.recognized` - Face recognized
- `scan.completed` - Scan completed
- `model.trained` - Model training completed
- `alert.triggered` - Alert triggered

### Webhook Configuration
```json
{
  "url": "https://your-app.com/webhooks/face-scan",
  "events": ["face.detected", "face.recognized"],
  "secret": "your-webhook-secret"
}
```

## Examples

### Basic Face Detection
```bash
curl -X POST "http://localhost:5000/api/v1/faces/detect" \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@image.jpg"
```

### Face Recognition
```bash
curl -X POST "http://localhost:5000/api/v1/faces/recognize" \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@image.jpg"
```

### Batch Processing
```bash
curl -X POST "http://localhost:5000/api/v1/scan/batch" \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["image1.jpg", "image2.jpg", "image3.jpg"],
    "options": {
      "detect_faces": true,
      "recognize_faces": true,
      "return_embeddings": false
    }
  }'
```

## Testing

### Postman Collection
Import the Postman collection from `docs/api/postman/face-scan-api.json` for easy API testing.

### API Testing Script
```python
import requests
import json

# Test API endpoints
def test_api():
    base_url = "http://localhost:5000/api/v1"
    
    # Test health endpoint
    response = requests.get(f"{base_url}/health")
    print(f"Health check: {response.status_code}")
    
    # Test authentication
    auth_data = {"username": "admin", "password": "admin123"}
    response = requests.post(f"{base_url}/auth/login", json=auth_data)
    token = response.json()["data"]["access_token"]
    
    # Test face detection
    headers = {"Authorization": f"Bearer {token}"}
    with open("test_image.jpg", "rb") as f:
        files = {"image": f}
        response = requests.post(f"{base_url}/faces/detect", headers=headers, files=files)
        print(f"Face detection: {response.status_code}")

if __name__ == "__main__":
    test_api()
```

## Support

For API support and questions:
- Email: support@facescanproject.com
- Documentation: https://docs.facescanproject.com
- GitHub Issues: https://github.com/yourusername/face-scan-project/issues
