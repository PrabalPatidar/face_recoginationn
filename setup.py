#!/usr/bin/env python3
"""
Setup script for Face Recognition System
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} detected")
    return True

def install_dependencies():
    """Install Python dependencies."""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    return run_command("pip install -r requirements.txt", "Installing Python dependencies")

def setup_environment():
    """Set up environment file."""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if not env_example.exists():
        print("❌ env.example not found")
        return False
    
    # Copy env.example to .env
    try:
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        print("✅ Created .env file from env.example")
        print("⚠️  Please edit .env file with your Supabase credentials")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def check_supabase_schema():
    """Check if Supabase schema file exists."""
    schema_file = Path("supabase_schema.sql")
    if not schema_file.exists():
        print("❌ supabase_schema.sql not found")
        return False
    
    print("✅ Supabase schema file found")
    print("⚠️  Remember to run the SQL schema in your Supabase project")
    return True

def main():
    """Main setup function."""
    print("🔧 Face Recognition System Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed at dependency installation")
        sys.exit(1)
    
    # Setup environment
    if not setup_environment():
        print("\n❌ Setup failed at environment setup")
        sys.exit(1)
    
    # Check Supabase schema
    if not check_supabase_schema():
        print("\n❌ Setup failed at schema check")
        sys.exit(1)
    
    print("\n" + "=" * 40)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file with your Supabase credentials")
    print("2. Run the SQL schema in your Supabase project")
    print("3. Run: python run.py")
    print("\n📖 For detailed instructions, see README_FACE_RECOGNITION.md")

if __name__ == "__main__":
    main()