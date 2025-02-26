import uvicorn
from asgiref.wsgi import WsgiToAsgi
from server import app  # Import your Flask app

# Wrap the Flask app with WsgiToAsgi
asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    uvicorn.run(asgi_app, host="0.0.0.0", port=5000)
