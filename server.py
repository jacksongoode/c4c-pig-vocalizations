from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Directory to save uploads:
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dummy function to process audio file. Replace this with your model prediction logic.
def process_audio(file_path):
    # Here you can incorporate your audio processing pipeline, e.g., computing a spectrogram and running your model.
    # For demo purposes, we return a dummy JSON result.
    result = {
        "message": "Audio processed successfully",
        "file": os.path.basename(file_path),
        "prediction": "Dummy classification"
    }
    return result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'audio-file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['audio-file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    # Secure the filename and save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Process the file (call your model here)
    result = process_audio(file_path)

    # Optionally, delete the file after processing
    # os.remove(file_path)

    return jsonify(result)


if __name__ == '__main__':
    app.run()