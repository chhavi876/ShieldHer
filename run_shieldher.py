
import subprocess
import sys
import os
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit', 'plotly', 'pandas', 'numpy', 
        'cv2', 'speech_recognition', 'PIL', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install missing packages using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def create_data_directory():
    """Create necessary directories"""
    data_dir = Path("data")
    recordings_dir = Path("recordings")
    
    data_dir.mkdir(exist_ok=True)
    recordings_dir.mkdir(exist_ok=True)
    
    print("✅ Created necessary directories!")

def launch_streamlit():
    """Launch the Streamlit application"""
    print("🚀 Starting ShieldHer application...")
    
    # Streamlit configuration
    config_args = [
        "--server.port=8501",
        "--server.address=localhost",
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
        "--server.runOnSave=true",
        "--theme.base=light"
    ]
    
    # Build command
    cmd = [sys.executable, "-m", "streamlit", "run", "shieldher_app.py"] + config_args
    
    try:
        # Launch Streamlit
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Open browser
        webbrowser.open("http://localhost:8501")
        
        print("✅ ShieldHer is now running!")
        print("🌐 Open your browser to: http://localhost:8501")
        print("⚠️  Press Ctrl+C to stop the application")
        
        # Wait for process to complete
        process.wait()
        
    except KeyboardInterrupt:
        print("\n🛑 ShieldHer application stopped.")
        process.terminate()
    except Exception as e:
        print(f"❌ Error launching application: {e}")

def main():
    """Main startup function"""
    print("=" * 50)
    print("🛡️  ShieldHer - Personal Safety Monitor")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create directories
    create_data_directory()
    
    # Launch application
    launch_streamlit()

if __name__ == "__main__":
    main()