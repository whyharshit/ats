#!/usr/bin/env python3
"""
Setup script for ATS Expert
This script helps users set up the project dependencies and environment.
"""

import os
import sys
import subprocess
import shutil

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_pip():
    """Check if pip is available."""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                      check=True, capture_output=True)
        print("✅ pip is available")
        return True
    except subprocess.CalledProcessError:
        print("❌ pip is not available")
        return False

def install_requirements():
    """Install required packages."""
    try:
        print("📦 Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_env_file():
    """Check if .env file exists and has required variables."""
    if not os.path.exists(".env"):
        print("⚠️  .env file not found")
        print("📝 Creating .env file from .env.example...")
        if os.path.exists(".env.example"):
            shutil.copy(".env.example", ".env")
            print("✅ .env file created from .env.example")
            print("⚠️  Please update .env file with your actual API key")
        else:
            print("❌ .env.example not found")
            return False
    else:
        print("✅ .env file exists")
    
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up ATS Expert...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check pip
    if not check_pip():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check environment file
    if not check_env_file():
        sys.exit(1)
    
    print("=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Update your .env file with your Google API key")
    print("2. Install Poppler for Windows if not already installed")
    print("3. Run: streamlit run ATS.py")
    print("\n📖 For more information, see README.md")

if __name__ == "__main__":
    main() 