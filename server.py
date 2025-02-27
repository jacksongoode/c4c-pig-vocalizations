import os

import librosa
import matplotlib
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# Check if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

app = Flask(__name__)

# Directory to save uploads:
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Global variables to hold the loaded models – they will be loaded once
MODEL_VAL = None
MODEL_CON = None

# Pre-create the image transform so it's not re-instantiated on every call
PREPROCESS_TRANSFORM = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()]
)


def load_models():
    print("Loading models...")
    global MODEL_VAL, MODEL_CON
    # Define checkpoint paths (adjust as needed)
    CHECKPOINT_PATH_VAL = os.path.join(os.getcwd(), "checkpoints", "Valence", "save", "model_checkpoint_best.pt")
    CHECKPOINT_PATH_CON = os.path.join(os.getcwd(), "checkpoints", "Context", "save", "model_checkpoint_best.pt")

    if os.path.exists(CHECKPOINT_PATH_VAL):
        try:
            MODEL_VAL = load_resnet50_model(2, CHECKPOINT_PATH_VAL)
            print("Loaded valence model from checkpoint:", CHECKPOINT_PATH_VAL)
        except Exception as e:
            print("Error loading valence model:", e)
    else:
        print("Valence model checkpoint not found.")

    if os.path.exists(CHECKPOINT_PATH_CON):
        try:
            MODEL_CON = load_resnet50_model(18, CHECKPOINT_PATH_CON)
            print("Loaded context model from checkpoint:", CHECKPOINT_PATH_CON)
        except Exception as e:
            print("Error loading context model:", e)
    else:
        print("Context model checkpoint not found.")


def load_resnet50_model(num_classes, checkpoint_path):
    # Create a ResNet50 model, modify the final fc layer for num_classes
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    # Load state dict from checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


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
        x_padded = np.pad(x, (pad_before, pad_after), mode="constant")
    else:
        raise ValueError(f"Audio duration exceeds {max_duration} seconds.")

    # Compute spectrogram using STFT
    window_length = 512
    nfft = 512
    noverlap = int(0.99 * window_length)  # 99% overlap
    hop_length = window_length - noverlap
    S = librosa.stft(
        x_padded,
        n_fft=nfft,
        hop_length=hop_length,
        win_length=window_length,
        window="hann",
    )
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    # Save color spectrogram image with a longer height
    spectrogram_path = file_path + "_spec.png"
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log", cmap="viridis"
    )
    plt.ylim(0, 8000)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(spectrogram_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Load the image and preprocess for PyTorch using the pre-instantiated transform
    image = Image.open(spectrogram_path).convert("RGB")
    input_tensor = PREPROCESS_TRANSFORM(image).unsqueeze(0).to(device)

    return input_tensor, spectrogram_path


# Helper function to get context mapping from CSV or generate it from Excel if CSV does not exist
def get_context_mapping():
    mapping_file = "context_mapping.csv"
    if os.path.exists(mapping_file):
        df = pd.read_csv(mapping_file)
        mapping = {int(row["id"]): row["context"] for index, row in df.iterrows()}
    else:
        excel_file = "SoundwelDatasetKey.xlsx"
        data = pd.read_excel(excel_file)
        unique_contexts = sorted(set(data["Context"].tolist()))
        mapping = {i: context for i, context in enumerate(unique_contexts)}
        pd.DataFrame(list(mapping.items()), columns=["id", "context"]).to_csv(
            mapping_file, index=False
        )
    return mapping


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "audio-file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["audio-file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Secure the filename and save the file
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Preprocess the file
    preprocessed_input, spectrogram_path = preprocess_audio(file_path)
    print("Preprocessed input shape:", preprocessed_input.shape)

    result = {
        "message": "Audio processed successfully",
        "file": os.path.basename(file_path),
        "spectrogram": spectrogram_path,
    }

    # Use the pre-loaded models rather than loading on every request
    if MODEL_VAL is not None:
        with torch.no_grad():
            val_prediction = MODEL_VAL(preprocessed_input)
            print("Raw valence prediction values:", val_prediction.cpu().numpy())
            predicted_val_class = int(torch.argmax(val_prediction, dim=1)[0])
            result["valence_prediction"] = predicted_val_class
    else:
        result["valence_prediction"] = "N/A"

    if MODEL_CON is not None:
        with torch.no_grad():
            con_prediction = MODEL_CON(preprocessed_input)
            print("Raw context prediction values:", con_prediction.cpu().numpy())
            predicted_con_class = int(torch.argmax(con_prediction, dim=1)[0])
            context_mapping = get_context_mapping()
            result["context_prediction"] = context_mapping.get(
                predicted_con_class, "Unknown"
            )
    else:
        result["context_prediction"] = "N/A"

    return jsonify(result)


# Serve static files from the uploads directory
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    load_models()
    app.run()
