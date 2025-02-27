import os
# Use PyTorch with MPS (Metal Performance Shaders) for Apple Silicon
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from collections import Counter
from sklearn.model_selection import train_test_split
from PIL import Image

# Check if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Set a default image size for ResNet50
IMAGE_SIZE = (224, 224)


class ImageDataset(Dataset):
    """PyTorch Dataset for loading images from file paths."""
    def __init__(self, file_paths, labels=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image


def balance_dataset(file_paths, labels):
    """Balance dataset by downsampling to the minimum class count."""
    # Count each label
    counter = Counter(labels)
    min_count = min(counter.values())
    balanced_files = []
    balanced_labels = []
    label_to_files = {}
    for fp, lab in zip(file_paths, labels):
        label_to_files.setdefault(lab, []).append(fp)
    for lab, fps in label_to_files.items():
        fps = np.array(fps)
        # Randomly choose min_count files for this label
        indices = np.random.choice(len(fps), min_count, replace=False)
        balanced_files.extend(fps[indices].tolist())
        balanced_labels.extend([lab] * min_count)
    return balanced_files, balanced_labels


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def nn_function(file_paths, labels, equalize_labels, minibatch_size, validation_patience, checkpoint_path):
    """
    Equivalent to MATLAB NN_function.m using PyTorch
    Parameters:
        file_paths (list): List of image file paths (strings).
        labels (list): List of labels corresponding to each image.
        equalize_labels (bool): Whether to equalize label counts in the dataset.
        minibatch_size (int): Batch size for training.
        validation_patience (int): Patience for early stopping.
        checkpoint_path (str): Directory to save model checkpoints.

    Returns:
        model: The trained PyTorch model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        val_labels: List of validation labels.
        val_label_counts: Dictionary with counts per label in the validation set.
        val_loader: Augmented validation data loader (same as val_loader here).
    """
    # If equalizing labels, balance the dataset
    if equalize_labels:
        file_paths, labels = balance_dataset(file_paths, labels)

    # Split dataset into training (70%) and validation (30%) sets
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.3, stratify=labels, random_state=42
    )

    # Create PyTorch datasets and data loaders
    train_dataset = ImageDataset(train_files, train_labels)
    val_dataset = ImageDataset(val_files, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=minibatch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=minibatch_size, shuffle=False)

    # Determine number of classes
    classes = np.unique(labels)
    num_classes = len(classes)

    # Load the pretrained ResNet50 model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze the base model
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classification layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Move model to device (CPU/GPU/MPS)
    model = model.to(device)

    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Prepare checkpoint directory
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=validation_patience,
        verbose=True,
        path=os.path.join(checkpoint_path, 'model_checkpoint_best.pt')
    )

    # Training loop
    n_epochs = 20
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, target_labels in train_loader:
            inputs, target_labels = inputs.to(device), target_labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, target_labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += target_labels.size(0)
            correct += (predicted == target_labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, target_labels in val_loader:
                inputs, target_labels = inputs.to(device), target_labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, target_labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += target_labels.size(0)
                val_correct += (predicted == target_labels).sum().item()

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total

        print(f'Epoch {epoch+1}/{n_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

        # Save checkpoint for this epoch
        torch.save(model.state_dict(), os.path.join(checkpoint_path, f'model_checkpoint_{epoch+1:02d}.pt'))

        # Early stopping
        early_stopping(val_epoch_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load the best model
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'model_checkpoint_best.pt')))

    return model, train_loader, val_loader, val_labels, dict(Counter(val_labels)), val_loader


if __name__ == '__main__':
    # Example usage
    # These paths and labels should be replaced with actual data
    file_paths = [os.path.join('data', f) for f in os.listdir('data') if f.endswith('.jpg') or f.endswith('.png')]
    labels = [0 if 'class0' in f else 1 for f in file_paths]  # dummy labeling logic

    model, train_loader, val_loader, val_labels, val_label_counts, aug_loader = nn_function(
        file_paths, labels, equalize_labels=True, minibatch_size=32, validation_patience=5, checkpoint_path='checkpoints'
    )
    print('Training complete.')