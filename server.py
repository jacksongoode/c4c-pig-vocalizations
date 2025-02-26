from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import tensorflow as tf  # added import for tensorflow

app = Flask(__name__)

# Directory to save uploads:
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model checkpoint if available
CHECKPOINT_PATH = os.path.join(os.getcwd(), 'checkpoints', 'Valence', 'Iter1', 'model_checkpoint_01.h5')
if os.path.exists(CHECKPOINT_PATH):
    try:
        model = tf.keras.models.load_model(CHECKPOINT_PATH)
        print('Loaded model from checkpoint:', CHECKPOINT_PATH)
    except Exception as e:
        print('Error loading model:', e)
        model = None
else:
    model = None

# Updated dummy function to process audio file using the loaded model if available
def process_audio(file_path):
    # Here you can incorporate your audio processing pipeline
    # For now, if the model is loaded, simulate a prediction response
    if model is not None:
        prediction = 'Prediction from checkpoint model'
    else:
        prediction = 'Dummy classification'
    result = {
        "message": "Audio processed successfully",
        "file": os.path.basename(file_path),
        "prediction": prediction
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