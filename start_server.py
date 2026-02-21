import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import uvicorn
    from api.main import app
    
    print("🚀 Starting Localization API Server...")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print("\nPress CTRL+C to stop the server\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)