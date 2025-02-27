import uvicorn
from asgiref.wsgi import WsgiToAsgi
from server import app, load_models

# Wrap the Flask app with WsgiToAsgi
asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    load_models()  # Ensure models are loaded
    uvicorn.run(asgi_app, host="0.0.0.0", port=5000)
