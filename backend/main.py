"""
SignBridge AI - Backend Entrypoint Module
Allows running `uvicorn main:app` directly from inside the backend/ directory.
"""

from app.main import app

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"[SignBridge FastAPI] Starting server on {host}:{port}...")
    uvicorn.run(app, host=host, port=port)
