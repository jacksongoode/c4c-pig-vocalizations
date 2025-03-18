import os
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict

import requests

# Get your Vercel Blob token from environment variables
BLOB_READ_WRITE_TOKEN = os.environ.get("BLOB_READ_WRITE_TOKEN")


def handler(request: BaseHTTPRequestHandler) -> Dict[str, Any]:
    """Handle file upload to Vercel Blob."""
    if request.method != "POST":
        return {"error": "Method not allowed"}, 405

    try:
        # Get the file from the request
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("multipart/form-data"):
            return {"error": "Content type must be multipart/form-data"}, 400

        # Parse the multipart form data
        boundary = content_type.split("=")[1].encode()
        remaining_bytes = int(request.headers.get("content-length", 0))

        # Skip headers to get to file content
        line = request.rfile.readline()
        remaining_bytes -= len(line)
        while line.strip(b"\r\n"):
            line = request.rfile.readline()
            remaining_bytes -= len(line)

        # Set up headers for Vercel Blob
        headers = {
            "Authorization": f"Bearer {BLOB_READ_WRITE_TOKEN}",
            "Content-Type": "application/octet-stream",
        }

        # Create a generator to stream the file content directly
        def file_content_generator():
            nonlocal remaining_bytes
            while remaining_bytes > 0:
                chunk = request.rfile.readline()
                remaining_bytes -= len(chunk)
                if boundary in chunk:
                    break
                yield chunk

        # Stream upload to Vercel Blob without loading entire file into memory
        response = requests.put(
            "https://api.vercel.com/v2/blobs",
            headers=headers,
            data=file_content_generator(),
        )

        if response.status_code != 200:
            return {"error": f"Failed to upload to Vercel Blob: {response.text}"}, 500

        blob_data = response.json()
        return {"url": blob_data["url"]}

    except Exception as e:
        return {"error": str(e)}, 500
