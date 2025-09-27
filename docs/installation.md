# Installation Guide

This guide will help you install and set up the Face Scan Project on your system.

## System Requirements

### Minimum Requirements
- Python 3.8 or higher
- 4GB RAM
- 2GB free disk space
- Webcam or camera device (for real-time scanning)

### Recommended Requirements
- Python 3.9 or higher
- 8GB RAM
- 5GB free disk space
- GPU with CUDA support (for faster processing)
- High-resolution camera

### Supported Operating Systems
- Windows 10/11
- macOS 10.15+
- Ubuntu 18.04+
- CentOS 7+

## Installation Methods

### Method 1: Using pip (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/face-scan-project.git
   cd face-scan-project
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package:**
   ```bash
   pip install -e .
   ```

### Method 2: Using Docker

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/face-scan-project.git
   cd face-scan-project
   ```

2. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

### Method 3: Manual Installation

1. **Install system dependencies:**

   **Ubuntu/Debian:**
   ```bash
   sudo apt-get update
   sudo apt-get install python3-dev python3-pip
   sudo apt-get install libopencv-dev python3-opencv
   sudo apt-get install libglib2.0-0 libsm6 libxext6 libxrender-dev
   ```

   **macOS:**
   ```bash
   brew install python3 opencv
   ```

   **Windows:**
   - Install Python from python.org
   - Install Visual Studio Build Tools
   - Install OpenCV from opencv.org

2. **Install Python dependencies:**
   ```bash
   pip install opencv-python
   pip install face-recognition
   pip install dlib
   pip install flask
   pip install numpy
   pip install pillow
   ```

## Configuration

### 1. Environment Setup

1. **Copy the environment template:**
   ```bash
   cp env.example .env
   ```

2. **Edit the `.env` file with your configuration:**
   ```bash
   # Database Configuration
   DATABASE_URL=postgresql://username:password@localhost:5432/face_scan_db
   
   # Application Configuration
   FLASK_ENV=development
   SECRET_KEY=your-secret-key-here
   
   # Face Recognition Configuration
   FACE_DETECTION_MODEL=hog
   CONFIDENCE_THRESHOLD=0.6
   ```

### 2. Database Setup

**Option A: SQLite (Default - No setup required)**
- SQLite database will be created automatically

**Option B: PostgreSQL**
1. Install PostgreSQL
2. Create a database:
   ```sql
   CREATE DATABASE face_scan_db;
   CREATE USER face_scan_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE face_scan_db TO face_scan_user;
   ```
3. Update `DATABASE_URL` in `.env` file

### 3. Model Setup

1. **Download pre-trained models:**
   ```bash
   python scripts/download_models.py
   ```

2. **Verify model installation:**
   ```bash
   python -c "import face_recognition; print('Models loaded successfully')"
   ```

## Initial Setup

### 1. Run Database Migrations
```bash
python scripts/setup_environment.py
```

### 2. Create Initial Data
```bash
python scripts/data_preprocessing.py
```

### 3. Test Installation
```bash
python -m pytest tests/ -v
```

## Running the Application

### Development Mode
```bash
python src/face_scan/app.py
```

### Production Mode
```bash
gunicorn -w 4 -b 0.0.0.0:5000 src.face_scan.app:app
```

### GUI Mode
```bash
python src/face_scan/main.py
```

## Verification

### 1. Health Check
Visit `http://localhost:5000/health` to verify the API is running.

### 2. Test Face Detection
```bash
curl -X POST -F "image=@tests/sample_image.jpg" \
  http://localhost:5000/api/v1/scan/detect
```

### 3. Check Logs
```bash
tail -f logs/app.log
```

## Troubleshooting

### Common Issues

**1. ImportError: No module named 'cv2'**
```bash
pip install opencv-python
```

**2. dlib installation fails**
```bash
# On Ubuntu/Debian
sudo apt-get install cmake
pip install dlib

# On macOS
brew install cmake
pip install dlib
```

**3. face_recognition installation fails**
```bash
# Install dlib first, then face_recognition
pip install dlib
pip install face_recognition
```

**4. Camera not detected**
- Check camera permissions
- Verify camera is not being used by another application
- Try different camera index in configuration

**5. Database connection errors**
- Verify database is running
- Check connection string in `.env` file
- Ensure database user has proper permissions

### Performance Issues

**1. Slow face detection**
- Use GPU acceleration if available
- Reduce image resolution
- Use HOG model instead of CNN for faster processing

**2. High memory usage**
- Reduce batch size in configuration
- Process images in smaller chunks
- Use image compression

### Getting Help

1. **Check the logs:**
   ```bash
   cat logs/error.log
   ```

2. **Run diagnostics:**
   ```bash
   python scripts/diagnostics.py
   ```

3. **Search issues:**
   - GitHub Issues: https://github.com/yourusername/face-scan-project/issues
   - Documentation: https://docs.facescan.com

## Uninstallation

### Remove Python Package
```bash
pip uninstall face-scan-project
```

### Remove Docker Containers
```bash
docker-compose down
docker rmi face_scan_project_app
```

### Clean Up Files
```bash
rm -rf face_scan_project/
rm -rf ~/.face_scan/
```

## Next Steps

After successful installation:

1. Read the [API Documentation](api.md)
2. Check out [Model Training Guide](model_training.md)
3. Review [Deployment Guide](deployment.md)
4. Explore the example notebooks in the `notebooks/` directory

## Support

For installation support:
- Email: support@facescan.com
- GitHub Issues: https://github.com/yourusername/face-scan-project/issues
- Documentation: https://docs.facescan.com
