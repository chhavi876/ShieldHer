import os
import subprocess
import sys
from pathlib import Path

def clean_build():
    """Clean previous build files"""
    build_dirs = ['build', 'dist', '__pycache__']
    spec_files = ['*.spec']
    
    for dir_name in build_dirs:
        if os.path.exists(dir_name):
            subprocess.run(['rmdir', '/S', '/Q', dir_name], shell=True)
            print(f"✅ Cleaned {dir_name}")
    
    # Remove spec files
    for spec_file in Path('.').glob('*.spec'):
        spec_file.unlink()
        print(f"✅ Removed {spec_file}")

def build_executable():
    """Build the executable using PyInstaller"""
    print("🔨 Building ShieldHer executable...")
    
    # PyInstaller command
    cmd = [
        'pyinstaller',
        '--onedir',  # Create directory with dependencies
        '--windowed',  # Remove console window (optional)
        '--name=ShieldHer',
        '--icon=shieldher_icon.ico',  # Add icon if available
        '--add-data=data;data',  # Include data directory
        '--hidden-import=streamlit',
        '--hidden-import=plotly',
        '--hidden-import=pandas',
        '--hidden-import=numpy',
        '--hidden-import=cv2',
        '--hidden-import=speech_recognition',
        '--hidden-import=PIL',
        'shieldher_app.py'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Build completed successfully!")
        print("📁 Executable created in: dist/ShieldHer/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_installer_batch():
    """Create a batch file for easy installation"""
    batch_content = '''@echo off
echo Installing ShieldHer dependencies...
pip install -r requirements.txt
echo.
echo Dependencies installed successfully!
echo.
echo To run ShieldHer:
echo 1. Double-click run_shieldher.py
echo 2. Or run: python run_shieldher.py
echo.
pause
'''
    
    with open('install.bat', 'w') as f:
        f.write(batch_content)
    
    print("✅ Created install.bat file")

def main():
    """Main build function"""
    print("=" * 50)
    print("🔧 ShieldHer Build System")
    print("=" * 50)
    
    # Clean previous builds
    clean_build()
    
    # Create installer batch file
    create_installer_batch()
    
    # Build executable
    if build_executable():
        print("\n🎉 Build completed successfully!")
        print("📦 Find your executable in: dist/ShieldHer/ShieldHer.exe")
    else:
        print("\n❌ Build failed. Check the error messages above.")

if __name__ == "__main__":
    main()