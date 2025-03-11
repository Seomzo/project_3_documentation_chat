import subprocess
import sys
import os

def install_playwright_browsers():
    """Install Playwright browsers if they're not already installed."""
    try:
        # Check if we're running in a Streamlit Cloud environment
        if os.environ.get('STREAMLIT_SHARING', '') or os.path.exists('/home/appuser'):
            print("Running in Streamlit Cloud, installing Playwright browsers...")
            subprocess.run([sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"], 
                           check=True, capture_output=True)
            print("Playwright browsers installed successfully.")
        else:
            print("Not running in Streamlit Cloud, skipping automatic browser installation.")
    except Exception as e:
        print(f"Error installing Playwright browsers: {str(e)}")
        
if __name__ == "__main__":
    install_playwright_browsers() 