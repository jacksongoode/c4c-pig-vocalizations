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
        line = request.rfile.readline()
        remaining_bytes -= len(line)

        # Skip headers
        while line.strip(b"\r\n"):
            line = request.rfile.readline()
            remaining_bytes -= len(line)

        # Read the file content
        file_content = b""
        while remaining_bytes > 0:
            line = request.rfile.readline()
            remaining_bytes -= len(line)
            if boundary in line:
                break
            file_content += line

        # Upload to Vercel Blob
        headers = {
            "Authorization": f"Bearer {BLOB_READ_WRITE_TOKEN}",
            "Content-Type": "application/octet-stream",
        }

        response = requests.put(
            "https://api.vercel.com/v2/blobs", headers=headers, data=file_content
        )

        if response.status_code != 200:
            return {"error": "Failed to upload to Vercel Blob"}, 500

        blob_data = response.json()
        return {"url": blob_data["url"]}

    except Exception as e:
        return {"error": str(e)}, 500
