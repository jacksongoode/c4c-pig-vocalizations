import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Check if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Set a default image size for ResNet50
IMAGE_SIZE = (224, 224)


# Helper function to classify a dataset using the PyTorch model
def classify_dataset(model, dataloader):
    """Return predicted labels for the dataset."""
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
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.transform = transforms.Compose(
            [transforms.Resize(IMAGE_SIZE), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image


def create_dataset(file_paths, batch_size=1):
    """Create a DataLoader from file paths."""
    dataset = PredictionDataset(file_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


if __name__ == "__main__":
    # Load the model from the checkpoint
    model_val = torch.load(
        "checkpoints/Valence/Iter1/model_checkpoint_best.pt", map_location=device
    )
    model_con = torch.load(
        "checkpoints/Context/Iter1/model_checkpoint_best.pt", map_location=device
    )

    # Set models to evaluation mode
    model_val.eval()
    model_con.eval()

    files = glob("test_soundwel/*.png")

    # Create the dataset
    dataloader = create_dataset(files)

    # Get predictions
    pred_val = classify_dataset(model_val, dataloader)
    pred_con = classify_dataset(model_con, dataloader)

    # Print out the predictions
    print("Valence")
    print(pred_val)
    print("Context")
    print(pred_con)
