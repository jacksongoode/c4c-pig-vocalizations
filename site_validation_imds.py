import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Check if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Set target image dimensions
IMAGE_SIZE = (224, 224)


class SiteValidationDataset(Dataset):
    """Dataset for site validation images."""
    def __init__(self, files, labels, base_dir="soundwel"):
        self.files = files
        self.labels = labels
        self.base_dir = base_dir
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_dir, self.files[idx])
        label = self.labels[idx]

        # Check if file exists and is an image
        is_image = img_path.lower().endswith(('.png', '.jpg', '.jpeg'))

        if os.path.exists(img_path) and is_image:
            try:
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            except Exception:
                # Return a blank image if there's an error
                image = torch.zeros((3, IMAGE_SIZE[0], IMAGE_SIZE[1]))
        else:
            # Return a blank image for non-images
            image = torch.zeros((3, IMAGE_SIZE[0], IMAGE_SIZE[1]))

        return image, label


def site_validation_imds(files, labels, base_dir="soundwel"):
    """
    Creates a DataLoader for site validation images.

    Parameters:
        files (list): List of file paths (strings). These are appended to the base_dir.
        labels (list): List of labels corresponding to each file.
        base_dir (str): Base directory path where the images are located.

    Returns:
        DataLoader: A DataLoader that yields batches of (image, label) pairs.
    """
    # Create dataset
    dataset = SiteValidationDataset(files, labels, base_dir)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    return dataloader


if __name__ == '__main__':
    # Example usage
    files = ["example1.jpg", "example2.jpg"]
    labels = [0, 1]
    dataloader = site_validation_imds(files, labels)
    # Get the first batch to check
    for images, labs in dataloader:
        print(images.shape, labs)
        break