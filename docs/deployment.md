# Deployment Guide

This guide covers deploying the Face Scan Project in various environments, from development to production.

## Deployment Overview

The Face Scan Project can be deployed in multiple ways:
- **Local Development**: Single machine setup
- **Docker**: Containerized deployment
- **Cloud**: AWS, Azure, GCP deployment
- **Kubernetes**: Scalable container orchestration
- **Edge**: IoT and embedded device deployment

## Prerequisites

### System Requirements
- Docker and Docker Compose (for containerized deployment)
- Kubernetes cluster (for K8s deployment)
- Cloud account (for cloud deployment)
- SSL certificate (for production HTTPS)

### Security Considerations
- API key management
- Database security
- Network security
- Data encryption
- Access control

## Local Development Deployment

### 1. Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/face-scan-project.git
cd face-scan-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp env.example .env
# Edit .env with your configuration

# Run application
python src/face_scan/app.py
```

### 2. Database Setup

**SQLite (Default):**
```bash
# No additional setup required
```

**PostgreSQL:**
```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb face_scan_db
sudo -u postgres createuser face_scan_user
sudo -u postgres psql -c "ALTER USER face_scan_user PASSWORD 'your_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE face_scan_db TO face_scan_user;"
```

### 3. Production Configuration

```bash
# Set production environment
export FLASK_ENV=production
export SECRET_KEY=your-secure-secret-key
export DATABASE_URL=postgresql://user:pass@localhost/face_scan_db

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 src.face_scan.app:app
```

## Docker Deployment

### 1. Single Container

```bash
# Build image
docker build -t face-scan-app .

# Run container
docker run -d \
  --name face-scan \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -e FLASK_ENV=production \
  face-scan-app
```

### 2. Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Scale application
docker-compose up -d --scale app=3
```

### 3. Multi-stage Build

```dockerfile
# Dockerfile.production
FROM python:3.9-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "src.face_scan.app:app"]
```

## Cloud Deployment

### AWS Deployment

#### 1. EC2 Instance

```bash
# Launch EC2 instance
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-group-ids sg-12345678 \
  --user-data file://user-data.sh
```

#### 2. ECS with Fargate

```yaml
# ecs-task-definition.json
{
  "family": "face-scan-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "face-scan-app",
      "image": "your-account.dkr.ecr.region.amazonaws.com/face-scan:latest",
      "portMappings": [
        {
          "containerPort": 5000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "FLASK_ENV",
          "value": "production"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/face-scan",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### 3. Lambda Function

```python
# lambda_handler.py
import json
from face_scan.api.routes.scan import detect_faces

def lambda_handler(event, context):
    try:
        # Process the request
        result = detect_faces(event)
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### Azure Deployment

#### 1. App Service

```bash
# Create App Service
az webapp create \
  --resource-group myResourceGroup \
  --plan myAppServicePlan \
  --name face-scan-app \
  --deployment-local-git

# Deploy code
git remote add azure https://face-scan-app.scm.azurewebsites.net/face-scan-app.git
git push azure main
```

#### 2. Container Instances

```bash
# Deploy container
az container create \
  --resource-group myResourceGroup \
  --name face-scan-container \
  --image your-registry.azurecr.io/face-scan:latest \
  --dns-name-label face-scan-app \
  --ports 5000
```

### Google Cloud Platform

#### 1. Cloud Run

```bash
# Deploy to Cloud Run
gcloud run deploy face-scan-app \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### 2. Compute Engine

```bash
# Create VM instance
gcloud compute instances create face-scan-vm \
  --image-family ubuntu-2004-lts \
  --image-project ubuntu-os-cloud \
  --machine-type e2-medium \
  --zone us-central1-a
```

## Kubernetes Deployment

### 1. Basic Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: face-scan-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: face-scan-app
  template:
    metadata:
      labels:
        app: face-scan-app
    spec:
      containers:
      - name: face-scan-app
        image: face-scan:latest
        ports:
        - containerPort: 5000
        env:
        - name: FLASK_ENV
          value: "production"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: face-scan-service
spec:
  selector:
    app: face-scan-app
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
```

### 2. Advanced Configuration

```yaml
# k8s-advanced.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: face-scan-app
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
  selector:
    matchLabels:
      app: face-scan-app
  template:
    metadata:
      labels:
        app: face-scan-app
    spec:
      containers:
      - name: face-scan-app
        image: face-scan:latest
        ports:
        - containerPort: 5000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: face-scan-secrets
              key: database-url
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: face-scan-service
spec:
  selector:
    app: face-scan-app
  ports:
  - port: 80
    targetPort: 5000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: face-scan-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: face-scan-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Edge Deployment

### 1. Raspberry Pi

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install python3-pip python3-opencv

# Install project
pip3 install -r requirements.txt

# Run application
python3 src/face_scan/app.py
```

### 2. NVIDIA Jetson

```bash
# Install JetPack
sudo apt-get update
sudo apt-get install python3-pip

# Install TensorFlow for Jetson
pip3 install tensorflow-gpu

# Run application
python3 src/face_scan/app.py
```

### 3. Mobile Deployment

```python
# Convert to TensorFlow Lite
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('face_recognition_model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save model
with open('face_recognition_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## Monitoring and Logging

### 1. Application Monitoring

```python
# monitoring.py
import logging
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
```

### 2. Health Checks

```python
# health_check.py
from flask import jsonify
import psutil
import time

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent
    })
```

### 3. Log Aggregation

```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  elasticsearch:
    image: elasticsearch:7.14.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"
  
  kibana:
    image: kibana:7.14.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
  
  logstash:
    image: logstash:7.14.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch
```

## Security Configuration

### 1. SSL/TLS Setup

```nginx
# nginx.conf
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 2. API Security

```python
# security.py
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per minute"]
)

@app.before_request
def before_request():
    # API key validation
    api_key = request.headers.get('Authorization')
    if not validate_api_key(api_key):
        return jsonify({'error': 'Invalid API key'}), 401
```

### 3. Database Security

```python
# database_security.py
import ssl

DATABASE_CONFIG = {
    'host': 'your-db-host',
    'port': 5432,
    'database': 'face_scan_db',
    'user': 'face_scan_user',
    'password': 'secure_password',
    'sslmode': 'require',
    'sslcert': '/path/to/client-cert.pem',
    'sslkey': '/path/to/client-key.pem',
    'sslrootcert': '/path/to/ca-cert.pem'
}
```

## Performance Optimization

### 1. Caching

```python
# caching.py
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'redis'})

@cache.memoize(timeout=300)
def get_face_encodings(image_path):
    # Expensive face encoding computation
    return face_encodings
```

### 2. Load Balancing

```nginx
# nginx-load-balancer.conf
upstream face_scan_backend {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
}

server {
    listen 80;
    location / {
        proxy_pass http://face_scan_backend;
    }
}
```

### 3. Database Optimization

```python
# database_optimization.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

## Backup and Recovery

### 1. Database Backup

```bash
# PostgreSQL backup
pg_dump -h localhost -U face_scan_user face_scan_db > backup_$(date +%Y%m%d).sql

# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h localhost -U face_scan_user face_scan_db | gzip > backups/backup_$DATE.sql.gz
```

### 2. Model Backup

```bash
# Backup models
tar -czf models_backup_$(date +%Y%m%d).tar.gz data/models/

# Upload to cloud storage
aws s3 cp models_backup_$(date +%Y%m%d).tar.gz s3://your-backup-bucket/
```

### 3. Disaster Recovery

```bash
# Recovery script
#!/bin/bash
# Restore database
gunzip -c backup_20231201.sql.gz | psql -h localhost -U face_scan_user face_scan_db

# Restore models
tar -xzf models_backup_20231201.tar.gz

# Restart services
docker-compose restart
```

## Troubleshooting

### Common Deployment Issues

**1. Port Conflicts**
```bash
# Check port usage
netstat -tulpn | grep :5000
lsof -i :5000
```

**2. Memory Issues**
```bash
# Monitor memory usage
free -h
docker stats
```

**3. Database Connection Issues**
```bash
# Test database connection
psql -h localhost -U face_scan_user -d face_scan_db -c "SELECT 1;"
```

**4. SSL Certificate Issues**
```bash
# Check certificate validity
openssl x509 -in certificate.crt -text -noout
```

## Maintenance

### 1. Regular Updates

```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Update Docker images
docker-compose pull
docker-compose up -d
```

### 2. Log Rotation

```bash
# Configure logrotate
cat > /etc/logrotate.d/face-scan << EOF
/var/log/face-scan/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 www-data www-data
}
EOF
```

### 3. Performance Monitoring

```bash
# Monitor application performance
curl -s http://localhost:5000/health | jq
docker stats --no-stream
```

## Support

For deployment support:
- Email: deployment@facescan.com
- GitHub Issues: https://github.com/yourusername/face-scan-project/issues
- Documentation: https://docs.facescan.com
