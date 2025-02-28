import os

import torch

# Use a file-based flag to track if device info has been printed
_FLAG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".device_printed")


def get_device() -> torch.device:
    """Return the best available device (CUDA, MPS, or CPU).

    Returns:
        torch.device: The best available device for PyTorch operations.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Print device info only once using a file-based flag
    if not os.path.exists(_FLAG_FILE):
        print(f"Using device: {device}")
        # Create the flag file to indicate we've printed the device info
        with open(_FLAG_FILE, "w") as f:
            f.write("1")

    return device


def setup_checkpoint_dir(checkpoint_path: str) -> str:
    """Create checkpoint directory if it doesn't exist.

    Args:
        checkpoint_path: Path to the checkpoint directory.

    Returns:
        str: The checkpoint path.
    """
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)
    return checkpoint_path


def load_model_from_checkpoint(
    model: torch.nn.Module, checkpoint_path: str
) -> torch.nn.Module:
    """Load model weights from checkpoint.

    Args:
        model: PyTorch model to load weights into.
        checkpoint_path: Path to the checkpoint file.

    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    device = get_device()
    if os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded model checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading model checkpoint: {e}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")

    return model.to(device)
