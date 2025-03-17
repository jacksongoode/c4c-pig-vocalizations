import functools
import json
import os
import tempfile
from typing import Dict, Tuple

import librosa
import matplotlib
import numpy as np
import requests
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request, send_from_directory
from torchvision import models, transforms
from werkzeug.utils import secure_filename

from utils import get_device

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Use the utility function to get the device
device = get_device()

app = Flask(__name__)

# Determine if we're running on Vercel based on environment variables
is_vercel = os.environ.get("VERCEL") == "1"

# Directory to save uploads - use /tmp on Vercel, uploads folder locally
UPLOAD_FOLDER = "/tmp" if is_vercel else "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Global variables to hold the loaded models – they will be loaded once
MODEL_VAL = None
MODEL_CON = None

# Pre-create the image transform so it's not re-instantiated on every call
PREPROCESS_TRANSFORM = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()]
)


# Add Vercel handler - this is used only in Vercel environment
def handler(request):
    """Handle requests from Vercel."""
    with app.request_context(request):
        return app.dispatch_request()


@functools.lru_cache(maxsize=2)
def get_model(model_type: str) -> nn.Module:
    """Load and cache model based on type.

    Args:
        model_type: Type of model to load ('valence' or 'context').

    Returns:
        nn.Module: Loaded model.

    Raises:
        ValueError: If model_type is not 'valence' or 'context'.
        FileNotFoundError: If model checkpoint is not found.
    """
    if model_type == "valence":
        checkpoint_path = os.path.join(
            os.getcwd(),
            "checkpoints",
            "Valence",
            "save",
            "model_checkpoint_best_acc.pt",
        )
        num_classes = 2
    elif model_type == "context":
        checkpoint_path = os.path.join(
            os.getcwd(),
            "checkpoints",
            "Context",
            "save",
            "model_checkpoint_best_acc.pt",
        )
        num_classes = 18
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if os.path.exists(checkpoint_path):
        model = load_efficientnet_model(num_classes, checkpoint_path)
        return model
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")


def load_models() -> None:
    """Load models into global variables."""
    print("Loading models...")
    global MODEL_VAL, MODEL_CON

    try:
        MODEL_VAL = get_model("valence")
        print("Loaded valence model successfully")
    except Exception as e:
        print(f"Error loading valence model: {e}")

    try:
        MODEL_CON = get_model("context")
        print("Loaded context model successfully")
    except Exception as e:
        print(f"Error loading context model: {e}")


def load_efficientnet_model(num_classes: int, checkpoint_path: str) -> nn.Module:
    """Create an EfficientNet model and load weights from checkpoint.

    Args:
        num_classes: Number of output classes.
        checkpoint_path: Path to the model checkpoint file.

    Returns:
        nn.Module: Loaded EfficientNet model.
    """
    # Create an EfficientNet model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Modify the final layer for num_classes
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

    # Load state dict from checkpoint
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(
            state_dict, strict=False
        )  # load with strict=False to ignore missing keys
        model.to(device)
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        print(f"Error loading model from {checkpoint_path}: {e}")
        raise


def preprocess_audio(file_path: str) -> Tuple[torch.Tensor, str]:
    """Process audio file to create spectrogram and prepare for model input.

    Args:
        file_path: Path to the audio file.

    Returns:
        Tuple[torch.Tensor, str]: Preprocessed input tensor and path to saved spectrogram image.

    Raises:
        ValueError: If audio duration exceeds max_duration.
    """
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
def get_context_mapping() -> Dict[int, str]:
    """Get mapping from context class IDs to human-readable labels.

    Returns:
        Dict[int, str]: Mapping from class IDs to context labels.
    """
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
    """Render the main page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Handle audio file upload and processing.

    Returns:
        JSON response with processing results or error message.
    """
    # Handle different request types based on environment
    is_json_request = (
        request.content_type and "application/json" in request.content_type
    )

    if is_json_request:
        # Handle blob URL format (Vercel deployment)
        data = request.get_json()
        if not data or "blobUrl" not in data:
            return jsonify({"error": "No blob URL provided"}), 400

        try:
            # Download the file from the blob URL
            response = requests.get(data["blobUrl"])
            if response.status_code != 200:
                return jsonify({"error": "Failed to download file from blob URL"}), 400

            # Create a temporary file
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(data.get("filename", "audio.wav"))[1],
            ) as temp_file:
                temp_file.write(response.content)
                file_path = temp_file.name

        except Exception as e:
            return jsonify({"error": f"Error downloading from blob: {str(e)}"}), 500
    else:
        # Handle direct file upload (local development)
        if "audio-file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["audio-file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        try:
            # Secure the filename and save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)
        except Exception as e:
            return jsonify({"error": f"Error saving file: {str(e)}"}), 500

    try:
        # Preprocess the file
        preprocessed_input, spectrogram_path = preprocess_audio(file_path)
        print("Preprocessed input shape:", preprocessed_input.shape)

        result = {
            "message": "Audio processed successfully",
            "file": os.path.basename(file_path),
        }

        # The way we handle spectrogram paths is different in local vs Vercel
        if is_vercel:
            # For Vercel, return the full path (the spectrogram is temporary)
            result["spectrogram"] = spectrogram_path
        else:
            # For local, return the path relative to the uploads folder
            result["spectrogram"] = f"/uploads/{os.path.basename(spectrogram_path)}"

        # Use the pre-loaded models rather than loading on every request
        if MODEL_VAL is not None:
            with torch.inference_mode():
                val_prediction = MODEL_VAL(preprocessed_input)
                print("Raw valence prediction values:", val_prediction.cpu().numpy())
                predicted_val_class = int(torch.argmax(val_prediction, dim=1)[0])
                result["valence_prediction"] = predicted_val_class
                confidence = torch.nn.functional.softmax(val_prediction, dim=1)[0][
                    predicted_val_class
                ].item()
                result["valence_confidence"] = f"{confidence:.2f}"
        else:
            result["valence_prediction"] = "N/A"
            result["valence_confidence"] = "N/A"

        if MODEL_CON is not None:
            with torch.inference_mode():
                con_prediction = MODEL_CON(preprocessed_input)
                print("Raw context prediction values:", con_prediction.cpu().numpy())
                predicted_con_class = int(torch.argmax(con_prediction, dim=1)[0])
                context_mapping = get_context_mapping()
                result["context_prediction"] = context_mapping.get(
                    predicted_con_class, "Unknown"
                )
                confidence = torch.nn.functional.softmax(con_prediction, dim=1)[0][
                    predicted_con_class
                ].item()
                result["context_confidence"] = f"{confidence:.2f}"
        else:
            result["context_prediction"] = "N/A"
            result["context_confidence"] = "N/A"

        # In Vercel environment, cleanup temporary files
        if is_vercel:
            os.unlink(file_path)
            os.unlink(spectrogram_path)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Serve static files from the uploads directory
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    """Serve uploaded files.

    Args:
        filename: Name of the file to serve.

    Returns:
        File response.
    """
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    load_models()  # Load models when running locally
    app.run(debug=False, host="0.0.0.0")
