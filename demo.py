#!/usr/bin/env python3
"""
Demo script for Face Recognition System
This script demonstrates how to use the API endpoints
"""

import requests
import json
import os
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api"

def test_health():
    """Test the health endpoint."""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure the application is running.")
        return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_recognize_face(image_path):
    """Test face recognition with an image."""
    print(f"🔍 Testing face recognition with {image_path}...")
    
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_BASE}/recognize", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Face recognition successful")
            print(f"   Result: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"❌ Face recognition failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Face recognition error: {e}")
        return False

def test_add_person(image_path, name, email=None, phone=None):
    """Test adding a new person."""
    print(f"👤 Testing add person: {name}...")
    
    if not Path(image_path).exists():
        print(f"❌ Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {'name': name}
            if email:
                data['email'] = email
            if phone:
                data['phone'] = phone
            
            response = requests.post(f"{API_BASE}/add-person", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Add person successful")
            print(f"   Result: {json.dumps(result, indent=2)}")
            return result
        else:
            print(f"❌ Add person failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Add person error: {e}")
        return False

def test_get_persons():
    """Test getting all persons."""
    print("👥 Testing get persons...")
    try:
        response = requests.get(f"{API_BASE}/persons")
        if response.status_code == 200:
            result = response.json()
            print("✅ Get persons successful")
            print(f"   Found {len(result)} persons")
            for person in result:
                print(f"   - {person['name']} ({person['email'] or 'No email'})")
            return result
        else:
            print(f"❌ Get persons failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Get persons error: {e}")
        return False

def main():
    """Main demo function."""
    print("🎭 Face Recognition System Demo")
    print("=" * 40)
    
    # Test health
    if not test_health():
        print("\n❌ Server is not running. Please start the application first:")
        print("   python run.py")
        return
    
    print("\n" + "-" * 40)
    
    # Test get persons (should be empty initially)
    test_get_persons()
    
    print("\n" + "-" * 40)
    
    # Look for test images
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(Path('.').glob(ext))
        test_images.extend(Path('.').glob(f'**/{ext}'))
    
    if not test_images:
        print("⚠️  No test images found in current directory")
        print("   Please add some image files (.jpg, .jpeg, .png) to test with")
        print("   Or modify this script to use specific image paths")
        return
    
    # Test with first image
    test_image = test_images[0]
    print(f"📸 Using test image: {test_image}")
    
    # Test recognition (should fail initially)
    result = test_recognize_face(test_image)
    
    if result and not result.get('is_match'):
        print("\n" + "-" * 40)
        # Add the person
        test_add_person(test_image, "Test Person", "test@example.com", "+1234567890")
        
        print("\n" + "-" * 40)
        # Test recognition again (should succeed now)
        test_recognize_face(test_image)
        
        print("\n" + "-" * 40)
        # Get all persons
        test_get_persons()
    
    print("\n" + "=" * 40)
    print("🎉 Demo completed!")
    print("\n💡 Tips:")
    print("- Add more images to test with different people")
    print("- Try the web interface at http://localhost:8000")
    print("- Check the Supabase dashboard to see stored data")

if __name__ == "__main__":
    main()
