import os
import sys

import uvicorn
from asgiref.wsgi import WsgiToAsgi
import requests

from server import app, load_models

# Wrap the Flask app with WsgiToAsgi
asgi_app = WsgiToAsgi(app)


def upload_model_to_blob(model_path):
    """Upload a model to Vercel Blob storage."""
    token = os.environ.get("BLOB_READ_WRITE_TOKEN")
    if not token:
        print("BLOB_READ_WRITE_TOKEN not found in environment variables")
        return None

    try:
        with open(model_path, "rb") as f:
            model_data = f.read()

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
        }

        response = requests.put(
            "https://api.vercel.com/v2/blobs",
            headers=headers,
            data=model_data
        )

        if response.status_code != 200:
            print(f"Failed to upload model to Vercel Blob: {response.text}")
            return None

        blob_data = response.json()
        print(f"Uploaded model to Vercel Blob: {blob_data['url']}")
        return blob_data["url"]
    except Exception as e:
        print(f"Error uploading model to Vercel Blob: {e}")
        return None


def main():
    """Main entry point for the application."""
    # Check if running on Vercel or local dev mode
    is_vercel = os.environ.get("VERCEL") == "1"
    is_dev = os.environ.get("DEV_MODE") == "1"
    
    if is_vercel:
        # On Vercel, upload models to Blob
        model_paths = [
            os.path.join("checkpoints", "Valence", "save", "model_checkpoint_best_acc.pt"),
            os.path.join("checkpoints", "Context", "save", "model_checkpoint_best_acc.pt")
        ]
        
        models_uploaded = True
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                url = upload_model_to_blob(model_path)
                if url:
                    # Store URL in environment for later retrieval
                    model_type = "VAL" if "Valence" in model_path else "CON"
                    os.environ[f"MODEL_{model_type}_URL"] = url
                else:
                    models_uploaded = False
            else:
                print(f"Model not found at {model_path}")
                models_uploaded = False
        
        if not models_uploaded:
            print("Some models could not be uploaded to Vercel Blob")
    
    # Try to load models, but continue even if they fail
    print("Loading models...")
    try:
        if is_dev:
            print("Running in development mode - skipping model loading")
            # In dev mode, don't try to load the actual models
            os.environ["SKIP_MODEL_LOADING"] = "1"
        load_models()
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Continuing without models - some functionality will be limited")

    try:
        # Get port from environment variable or use default
        port = int(os.environ.get("PORT", 5000))

        # Start the server
        print(f"Starting server on port {port}...")
        print(f"Access the application at: http://localhost:{port}")
        uvicorn.run(asgi_app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
