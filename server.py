from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def preprocess_audio(file_path):
    # Load audio file
    x, sr = librosa.load(file_path, sr=44100, mono=True)
    duration = len(x) / sr

    # Check duration and pad if necessary
    max_duration = 4.5
    if duration <= max_duration:
        total_length = int(max_duration * sr)
        current_length = len(x)
        pad_amount = total_length - current_length
        pad_before = pad_amount // 2
        pad_after = pad_amount - pad_before
        x_padded = np.pad(x, (pad_before, pad_after), mode='constant')
    else:
        raise ValueError(f"Audio duration exceeds {max_duration} seconds.")

    # Compute spectrogram using STFT
    window_duration = 0.03
    window_length = int(window_duration * sr)
    nfft = 2 ** int(np.ceil(np.log2(window_length)))  # Ensure n_fft is a power of 2 and >= window_length
    noverlap = int(0.99 * window_length)
    hop_length = window_length - noverlap
    S = librosa.stft(x_padded, n_fft=nfft, hop_length=hop_length, win_length=window_length, window='hann')
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    print("Shape of S_db before adding dimension:", S_db.shape)  # Debug: print shape before adding dimension
    S_db = S_db[:, :, np.newaxis]  # Add an extra dimension to make it 3D
    print("Shape of S_db after adding dimension:", S_db.shape)  # Debug: print shape after adding dimension

    # Save color spectrogram image with a longer height
    spectrogram_path = file_path + '_spec.png'
    plt.figure(figsize=(10, 4))  # Adjust the figure size for a longer height
    librosa.display.specshow(S_db[:, :, 0], sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap='viridis')
    plt.ylim(0, 8000)  # Limit y-axis to 0-8 kHz
    plt.axis('off')  # Turn off axes
    plt.tight_layout()
    plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Resize spectrogram to 224x224 using TensorFlow
    S_db = S_db.astype(np.float32)
    resized = tf.image.resize(S_db, [224, 224]).numpy()  # shape: (224,224)

    # Replicate the single channel to create 3-channel input
    three_channel = np.repeat(resized, 3, axis=-1)  # shape: (224,224,3)

    # Add batch dimension
    input_tensor = np.expand_dims(three_channel, axis=0)  # shape: (1,224,224,3)

    return input_tensor, spectrogram_path

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

    # Preprocess the file
    preprocessed_input, spectrogram_path = preprocess_audio(file_path)
    print("Preprocessed input shape:", preprocessed_input.shape)  # Debug: print input shape

    # Run prediction if model is loaded
    if model is not None:
        prediction = model.predict(preprocessed_input)
        print("Raw prediction values:", prediction)  # Debug: print raw output
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        result = {
            "message": "Audio processed successfully",
            "file": os.path.basename(file_path),
            "prediction": predicted_class,
            "spectrogram": spectrogram_path
        }
    else:
        result = {
            "message": "Model not loaded",
            "file": os.path.basename(file_path),
            "prediction": "N/A",
            "spectrogram": spectrogram_path
        }

    # Optionally, delete the file after processing
    # os.remove(file_path)

    return jsonify(result)

# Serve static files from the uploads directory
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run()