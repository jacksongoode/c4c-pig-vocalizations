import os
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

from utils import get_device, setup_checkpoint_dir

# Use the utility function to get the device
device = get_device()

# Training hyperparameters (matching MATLAB NN_function.m)
MAX_EPOCHS = 20
INITIAL_LEARNING_RATE = 0.001
MOMENTUM = 0.9
LEARN_RATE_DROP_PERIOD = 5
LEARN_RATE_DROP_FACTOR = 10 ** (-0.5)


class ImageDataset(Dataset):
    """PyTorch Dataset for loading images from file paths."""

    def __init__(self, file_paths: List[str], labels: Optional[List[int]] = None):
        """Initialize the dataset.

        Args:
            file_paths: List of paths to image files.
            labels: List of integer labels corresponding to each file.
        """
        self.file_paths = file_paths
        self.labels = labels
        # Use the official inference transforms from ResNet50_Weights.IMAGENET1K_V1
        self.transform = models.ResNet50_Weights.IMAGENET1K_V1.transforms()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, int]]:
                If labels are provided, returns (image, label) tuple.
                Otherwise, returns just the image tensor.
        """
        img_path = self.file_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if there's an error
            image = torch.zeros((3, 224, 224))

        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image


def balance_dataset(
    file_paths: List[str], labels: List[int]
) -> Tuple[List[str], List[int]]:
    """Balance dataset by downsampling to the minimum class count.

    Args:
        file_paths: List of paths to image files.
        labels: List of integer labels corresponding to each file.

    Returns:
        Tuple[List[str], List[int]]: Balanced file paths and labels.
    """
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

    def __init__(
        self,
        patience: int = 7,
        verbose: bool = False,
        delta: float = 0,
        path: str = "checkpoint.pt",
        progress_bar = None,
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait after last improvement.
            verbose: Whether to print messages.
            delta: Minimum change in monitored quantity to qualify as improvement.
            path: Path to save the checkpoint.
            progress_bar: tqdm progress bar to write messages to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.progress_bar = progress_bar

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        """Check if training should be stopped.

        Args:
            val_loss: Validation loss.
            model: PyTorch model to save if validation loss improves.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self._log_message(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module) -> None:
        """Save model checkpoint.

        Args:
            val_loss: Validation loss.
            model: PyTorch model to save.
        """
        if self.verbose:
            self._log_message(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def _log_message(self, message: str) -> None:
        """Log a message using the progress bar if available, otherwise print.

        Args:
            message: Message to log.
        """
        if self.progress_bar is not None:
            self.progress_bar.write(message)
        else:
            print(message)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    use_amp: bool = False,
) -> Tuple[float, float]:
    """Train the model for one epoch.

    Args:
        model: PyTorch model to train.
        train_loader: Training data loader.
        optimizer: Optimizer for updating model weights.
        criterion: Loss function.
        scaler: GradScaler for mixed precision training.
        use_amp: Whether to use mixed precision training.

    Returns:
        Tuple[float, float]: Training loss and accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Add progress bar for training - use position to prevent overlapping
    progress_bar = tqdm(train_loader, desc="Training", leave=False, position=1)

    for inputs, target_labels in progress_bar:
        inputs, target_labels = inputs.to(device), target_labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision if enabled
        try:
            # Try with device_type parameter (newer PyTorch versions)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, target_labels)
        except TypeError:
            # Fall back to older PyTorch versions or CPU
            if use_amp and device.type == 'cuda':
                # For older PyTorch with CUDA
                with torch.cuda.autocast(enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, target_labels)
            else:
                # For CPU or when AMP is disabled
                outputs = model(inputs)
                loss = criterion(outputs, target_labels)

        # Backward pass with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += target_labels.size(0)
        correct += (predicted == target_labels).sum().item()

        # Update progress bar with current loss and accuracy
        batch_loss = loss.item()
        batch_acc = (predicted == target_labels).sum().item() / target_labels.size(0)
        progress_bar.set_postfix({"loss": f"{batch_loss:.4f}", "acc": f"{batch_acc:.4f}"})

    return running_loss / total, correct / total


def validate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    use_amp: bool = False,
) -> Tuple[float, float]:
    """Evaluate the model on the validation set.

    Args:
        model: PyTorch model to evaluate.
        val_loader: Validation data loader.
        criterion: Loss function.
        use_amp: Whether to use mixed precision.

    Returns:
        Tuple[float, float]: Validation loss and accuracy.
    """
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # Add progress bar for validation - use position to prevent overlapping
    progress_bar = tqdm(val_loader, desc="Validating", leave=False, position=1)

    with torch.no_grad():
        for inputs, target_labels in progress_bar:
            inputs, target_labels = inputs.to(device), target_labels.to(device)

            # Use mixed precision if enabled
            try:
                # Try with device_type parameter (newer PyTorch versions)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, target_labels)
            except TypeError:
                # Fall back to older PyTorch versions or CPU
                if use_amp and device.type == 'cuda':
                    # For older PyTorch with CUDA
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = model(inputs)
                        loss = criterion(outputs, target_labels)
                else:
                    # For CPU or when AMP is disabled
                    outputs = model(inputs)
                    loss = criterion(outputs, target_labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += target_labels.size(0)
            val_correct += (predicted == target_labels).sum().item()

            # Update progress bar with current loss and accuracy
            batch_loss = loss.item()
            batch_acc = (predicted == target_labels).sum().item() / target_labels.size(0)
            progress_bar.set_postfix({"loss": f"{batch_loss:.4f}", "acc": f"{batch_acc:.4f}"})

    val_epoch_loss = val_loss / val_total
    val_epoch_acc = val_correct / val_total

    return val_epoch_loss, val_epoch_acc


def nn_function(
    file_paths: List[str],
    labels: List[int],
    equalize_labels: bool,
    minibatch_size: int,
    validation_patience: int,
    checkpoint_path: str,
    use_amp: bool = False,
    skip_training: bool = False,
) -> Tuple[nn.Module, DataLoader, DataLoader, List[int], Dict[int, int], DataLoader]:
    """
    Equivalent to MATLAB NN_function.m using PyTorch

    Args:
        file_paths: List of image file paths (strings).
        labels: List of labels corresponding to each image.
        equalize_labels: Whether to equalize label counts in the dataset.
        minibatch_size: Batch size for training.
        validation_patience: Patience for early stopping.
        checkpoint_path: Directory to save model checkpoints.
        use_amp: Whether to use mixed precision training.
        skip_training: Whether to skip training and load checkpoint if available.

    Returns:
        Tuple containing:
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=minibatch_size,
        shuffle=True,
        num_workers=2,  # Use multiple workers for parallel loading
        pin_memory=True,  # Speed up host to device transfers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=minibatch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Determine number of classes
    classes = np.unique(labels)
    num_classes = len(classes)

    # Initialize EfficientNet
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Modify the final layer to match the number of classes
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

    # Move model to device (CPU/GPU/MPS)
    model = model.to(device)

    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=INITIAL_LEARNING_RATE, momentum=MOMENTUM
    )

    # Set up learning rate scheduler - use StepLR to match MATLAB's piecewise scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LEARN_RATE_DROP_PERIOD,
        gamma=LEARN_RATE_DROP_FACTOR,
    )

    # Prepare checkpoint directory
    checkpoint_path = setup_checkpoint_dir(checkpoint_path)

    # Training loop
    n_epochs = MAX_EPOCHS

    # Create a progress bar for epochs - use position to prevent overlapping
    epoch_progress = tqdm(range(n_epochs), desc="Training Progress", position=0)

    # Setup early stopping with progress bar
    early_stopping = EarlyStopping(
        patience=validation_patience,
        verbose=True,
        path=os.path.join(checkpoint_path, "model_checkpoint_best.pt"),
        progress_bar=epoch_progress,
    )

    # Initialize GradScaler for mixed precision
    # Check if PyTorch version supports device_type parameter
    try:
        # Try with device_type parameter (newer PyTorch versions)
        scaler = torch.amp.GradScaler(device_type=device.type, enabled=use_amp)
    except TypeError:
        # Fall back to older PyTorch versions without device_type parameter
        scaler = torch.amp.GradScaler(enabled=use_amp)

    # If skip_training flag is True, load the checkpoint and skip training
    if skip_training:
        checkpoint_file = os.path.join(checkpoint_path, "model_checkpoint_best.pt")
        if os.path.exists(checkpoint_file):
            model.load_state_dict(torch.load(checkpoint_file, map_location=device))
            print(f"Loaded model checkpoint from {checkpoint_file}, skipping training.")
            lbl_counts = dict(Counter(val_labels))
            return model, train_loader, val_loader, val_labels, lbl_counts, val_loader
        else:
            print(f"Checkpoint not found at {checkpoint_file}. Training will proceed.")

    for epoch_idx in epoch_progress:
        # Train for one epoch
        epoch_loss, epoch_acc = train_model(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            use_amp,
        )

        # Validate after each epoch
        val_loss, val_acc = validate_model(model, val_loader, criterion, use_amp)

        # Update epoch progress bar with metrics
        epoch_progress.set_postfix({
            "epoch": epoch_idx + 1,
            "train_loss": f"{epoch_loss:.4f}",
            "train_acc": f"{epoch_acc:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "val_acc": f"{val_acc:.4f}"
        })

        # Step the learning rate scheduler
        scheduler.step()

        # Check early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load the best model
    model.load_state_dict(
        torch.load(
            os.path.join(checkpoint_path, "model_checkpoint_best.pt"),
            map_location=device,
        )
    )

    return (
        model,
        train_loader,
        val_loader,
        val_labels,
        dict(Counter(val_labels)),
        val_loader,
    )


if __name__ == "__main__":
    # Example usage
    # These paths and labels should be replaced with actual data
    file_paths = [
        os.path.join("data", f)
        for f in os.listdir("data")
        if f.endswith(".jpg") or f.endswith(".png")
    ]
    labels = [0 if "class0" in f else 1 for f in file_paths]  # dummy labeling logic

    model, train_loader, val_loader, val_labels, val_label_counts, aug_loader = (
        nn_function(
            file_paths,
            labels,
            equalize_labels=True,
            minibatch_size=32,
            validation_patience=5,
            checkpoint_path="checkpoints",
        )
    )
    print("Training complete.")
