#!/usr/bin/env python3
"""
Simple script to run the Face Recognition application
"""

import os
import sys
import uvicorn
from pathlib import Path

def check_requirements():
    """Check if required files and environment variables exist."""
    required_files = [
        "app.py",
        "requirements.txt",
        "templates/index.html"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    # Check environment variables
    if not os.getenv("SUPABASE_URL"):
        print("⚠️  SUPABASE_URL not set. Please create a .env file with your Supabase credentials.")
        print("   Copy env.example to .env and fill in your values.")
        return False
    
    if not os.getenv("SUPABASE_ANON_KEY"):
        print("⚠️  SUPABASE_ANON_KEY not set. Please create a .env file with your Supabase credentials.")
        print("   Copy env.example to .env and fill in your values.")
        return False
    
    return True

def main():
    """Main function to run the application."""
    print("🔍 Face Recognition System")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Setup incomplete. Please fix the issues above.")
        sys.exit(1)
    
    print("✅ All requirements met!")
    print("\n🚀 Starting Face Recognition application...")
    print("📱 Open your browser and go to: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 40)
    
    try:
        # Run the application
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
