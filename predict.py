from glob import glob
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from utils import get_device

# Use the utility function to get the device
device = get_device()

# Set a default image size for EfficientNet
IMAGE_SIZE = (224, 224)


# Helper function to classify a dataset using the PyTorch model
def classify_dataset(model: nn.Module, dataloader: DataLoader) -> np.ndarray:
    """Return predicted labels for the dataset.

    Args:
        model: PyTorch model to use for prediction.
        dataloader: DataLoader containing the input data.

    Returns:
        np.ndarray: Array of predicted class indices.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            # Forward pass
            outputs = model(inputs)
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())

    return np.array(all_preds)


# Create a dataset for prediction (no labels)
class PredictionDataset(Dataset):
    """Dataset for making predictions on images without labels.

    Attributes:
        file_paths: List of paths to image files.
        transform: Composition of image transformations to apply.
    """

    def __init__(self, file_paths: List[str]):
        """Initialize the dataset.

        Args:
            file_paths: List of paths to image files.
        """
        self.file_paths = file_paths
        self.transform = transforms.Compose(
            [transforms.Resize(IMAGE_SIZE), transforms.ToTensor()]
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            torch.Tensor: Processed image tensor.
        """
        img_path = self.file_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if there's an error
            image = torch.zeros((3, IMAGE_SIZE[0], IMAGE_SIZE[1]))
        return image


def create_dataset(file_paths: List[str], batch_size: int = 32) -> DataLoader:
    """Create a DataLoader from file paths.

    Args:
        file_paths: List of paths to image files.
        batch_size: Number of samples per batch.

    Returns:
        DataLoader: DataLoader for the prediction dataset.
    """
    dataset = PredictionDataset(file_paths)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,  # Use multiple workers for parallel loading
        pin_memory=True,  # Speed up host to device transfers
    )
    return dataloader


def load_efficientnet_model(num_classes: int, checkpoint_path: str) -> nn.Module:
    """Load an EfficientNet model with the specified number of output classes.

    Args:
        num_classes: Number of output classes.
        checkpoint_path: Path to the model checkpoint file.

    Returns:
        nn.Module: Loaded EfficientNet model.
    """
    # Create an EfficientNet model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Modify the final layer for the specified number of classes
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

    # Load the model weights
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading model from {checkpoint_path}: {e}")

    return model.to(device)


if __name__ == "__main__":
    # Load the models from the checkpoints
    try:
        model_val = load_efficientnet_model(
            2, "checkpoints/valence/save/model_checkpoint_best_acc.pt"
        )
        model_con = load_efficientnet_model(
            18, "checkpoints/context/save/model_checkpoint_best_acc.pt"
        )

        # Set models to evaluation mode
        model_val.eval()
        model_con.eval()

        # Get list of test files
        files = glob("test_soundwel/*.png")

        if not files:
            print("No test files found in test_soundwel directory.")
            exit(1)

        print(f"Found {len(files)} test files.")

        # Create the dataset with optimized batch size
        dataloader = create_dataset(files, batch_size=32)

        # Get predictions
        print("Generating valence predictions...")
        pred_val = classify_dataset(model_val, dataloader)

        print("Generating context predictions...")
        pred_con = classify_dataset(model_con, dataloader)

        # Print out the predictions
        print("\nResults:")
        print("Valence predictions:", pred_val)
        print("Context predictions:", pred_con)

        # Print summary statistics
        print("\nSummary:")
        print(f"Total files: {len(files)}")
        print(f"Valence distribution: {np.bincount(pred_val)}")
        print(f"Context distribution: {np.bincount(pred_con)}")

    except Exception as e:
        print(f"An error occurred: {e}")
