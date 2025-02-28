import os
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import get_device

# Use the utility function to get the device
device = get_device()

# Set target image dimensions
IMAGE_SIZE = (224, 224)


class SiteValidationDataset(Dataset):
    """Dataset for site validation images.

    This dataset loads images from disk and applies transformations for model input.
    It handles missing or corrupt images gracefully by returning blank tensors.

    Attributes:
        files: List of file paths relative to base_dir.
        labels: List of integer labels corresponding to each file.
        base_dir: Base directory containing the image files.
        transform: Composition of image transformations to apply.
    """

    def __init__(self, files: List[str], labels: List[int], base_dir: str = "soundwel"):
        """Initialize the dataset.

        Args:
            files: List of file paths relative to base_dir.
            labels: List of integer labels corresponding to each file.
            base_dir: Base directory containing the image files.
        """
        self.files = files
        self.labels = labels
        self.base_dir = base_dir
        self.transform = transforms.Compose(
            [transforms.Resize(IMAGE_SIZE), transforms.ToTensor()]
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            tuple: (image, label) where image is a tensor and label is an integer.
        """
        img_path = os.path.join(self.base_dir, self.files[idx])
        label = self.labels[idx]

        # Check if file exists and is an image
        is_image = img_path.lower().endswith((".png", ".jpg", ".jpeg"))

        if os.path.exists(img_path) and is_image:
            try:
                image = Image.open(img_path).convert("RGB")
                image = self.transform(image)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Return a blank image if there's an error
                image = torch.zeros((3, IMAGE_SIZE[0], IMAGE_SIZE[1]))
        else:
            # Return a blank image for non-images
            image = torch.zeros((3, IMAGE_SIZE[0], IMAGE_SIZE[1]))

        return image, label


def site_validation_imds(
    files: List[str], labels: List[int], base_dir: str = "soundwel"
) -> DataLoader:
    """
    Creates a DataLoader for site validation images.

    Args:
        files: List of file paths (strings). These are appended to the base_dir.
        labels: List of labels corresponding to each file.
        base_dir: Base directory path where the images are located.

    Returns:
        DataLoader: A DataLoader that yields batches of (image, label) pairs.
    """
    # Create dataset
    dataset = SiteValidationDataset(files, labels, base_dir)

    # Create dataloader with optimized settings
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,  # Use multiple workers for parallel loading
        pin_memory=True,  # Speed up host to device transfers
    )

    return dataloader


if __name__ == "__main__":
    # Example usage
    files = ["example1.jpg", "example2.jpg"]
    labels = [0, 1]
    dataloader = site_validation_imds(files, labels)
    # Get the first batch to check
    for images, labs in dataloader:
        print(f"Batch shape: {images.shape}, Labels: {labs}")
        break
