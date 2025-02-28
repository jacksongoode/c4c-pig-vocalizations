import os
import sys

import uvicorn
from asgiref.wsgi import WsgiToAsgi

from server import app, load_models

# Wrap the Flask app with WsgiToAsgi
asgi_app = WsgiToAsgi(app)


def main():
    """Main entry point for the application."""
    # Try to load models, but continue even if they fail
    print("Loading models...")
    try:
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
