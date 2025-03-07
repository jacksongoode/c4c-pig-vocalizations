import os
from collections import Counter
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
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


# Create a helper function for loading models from checkpoints
def load_model_from_checkpoint(model, checkpoint_path, device, strict=False, verbose=True):
    """Load a model from a checkpoint with proper error handling.

    Args:
        model: The model to load weights into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        strict: Whether to strictly enforce that the keys in state_dict match the model
        verbose: Whether to print detailed diagnostic messages

    Returns:
        Tuple[bool, str]: (Success status, Message)
    """
    if not os.path.exists(checkpoint_path):
        return False, f"Checkpoint not found: {checkpoint_path}"

    try:
        state_dict = torch.load(checkpoint_path, map_location=device)

        # If it's not a dict, it might be a complete model
        if not isinstance(state_dict, dict):
            return False, f"Invalid checkpoint format: {checkpoint_path} is not a state dictionary"

        # Check if keys match before loading
        model_keys = set(model.state_dict().keys())
        loaded_keys = set(state_dict.keys())

        missing_in_model = loaded_keys - model_keys
        missing_in_checkpoint = model_keys - loaded_keys

        # Load the state dict
        model.load_state_dict(state_dict, strict=strict)

        # Verify the model loaded correctly
        loaded_keys_count = len(set(model.state_dict().keys()).intersection(loaded_keys))
        total_keys = len(model_keys)
        if verbose:
            print(f"Loaded {loaded_keys_count}/{total_keys} model parameters")

        return True, f"Successfully loaded checkpoint: {checkpoint_path}"
    except Exception as e:
        return False, f"Error loading checkpoint: {e}"


class EarlyStopping:
    """Early stopping to stop training when validation loss or accuracy doesn't improve."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        progress_bar=None,
        monitor="loss",
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait after last improvement.
            verbose: Whether to print messages.
            delta: Minimum change in monitored quantity to qualify as improvement.
            path: Path to save the checkpoint.
            progress_bar: tqdm progress bar to write messages to.
            monitor: Metric to monitor ('loss' or 'accuracy').
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_acc_max = 0.0
        self.delta = delta
        self.path = path
        self.progress_bar = progress_bar
        self.monitor = monitor

        # Create paths for loss and accuracy checkpoints
        # Extract directory and filename
        directory = os.path.dirname(path)
        filename = os.path.basename(path)
        base_name = os.path.splitext(filename)[0]
        
        # Remove any existing 'best' from the base name
        base_name = base_name.replace('_best', '')
        
        # Create the paths with appropriate suffixes
        self.path = os.path.join(directory, f"{base_name}_best.pt")
        self.acc_path = os.path.join(directory, f"{base_name}_best_acc.pt")
        
        # Only log paths in verbose mode if explicitly requested
        if verbose and False:  # Disabled by default
            self._log_message(f"Checkpoint paths: \nLoss: {self.path}\nAccuracy: {self.acc_path}")

    def __call__(self, val_loss, val_acc, model):
        """Check if training should be stopped.

        Args:
            val_loss: Validation loss.
            val_acc: Validation accuracy.
            model: PyTorch model to save if validation metric improves.
        """
        # Always save best accuracy model regardless of monitor setting
        if val_acc > self.val_acc_max:
            self._save_checkpoint(
                model,
                self.acc_path,
                f"Validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f})",
            )
            self.val_acc_max = val_acc

        # Monitor either loss or accuracy for early stopping
        if self.monitor == "loss":
            score = -val_loss
            is_improvement = (
                score > self.best_score + self.delta
                if self.best_score is not None
                else True
            )

            if is_improvement:
                self._save_checkpoint(
                    model,
                    self.path,
                    f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f})",
                )
                self.val_loss_min = val_loss
                self.best_score = score
                self.counter = 0
            else:
                self._handle_no_improvement()

        else:  # monitor == 'accuracy'
            score = val_acc
            is_improvement = (
                score > self.best_score + self.delta
                if self.best_score is not None
                else True
            )

            if is_improvement:
                # Already saved in the always-save block above
                self.best_score = score
                self.counter = 0
            else:
                self._handle_no_improvement()

    def _handle_no_improvement(self):
        """Handle the case when there's no improvement in the monitored metric."""
        self.counter += 1
        if self.verbose:
            self._log_message(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
        if self.counter >= self.patience:
            self.early_stop = True

    def _save_checkpoint(self, model, path, message):
        """Save a model checkpoint.

        Args:
            model: The model to save
            path: Where to save the model
            message: Message to log on successful save
        """
        if self.verbose:
            self._log_message(message)

        try:
            torch.save(model.state_dict(), path)
            if self.verbose and os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                self._log_message(f"Saved checkpoint: {path} ({size_mb:.2f} MB)")
        except Exception as e:
            self._log_message(f"Error saving checkpoint: {e}")

    def _log_message(self, message):
        """Log a message using the progress bar if available, otherwise print."""
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
            if use_amp and device.type == "cuda":
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
        progress_bar.set_postfix(
            {"loss": f"{batch_loss:.4f}", "acc": f"{batch_acc:.4f}"}
        )

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
                if use_amp and device.type == "cuda":
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
            batch_acc = (predicted == target_labels).sum().item() / target_labels.size(
                0
            )
            progress_bar.set_postfix(
                {"loss": f"{batch_loss:.4f}", "acc": f"{batch_acc:.4f}"}
            )

    val_epoch_loss = val_loss / val_total
    val_epoch_acc = val_correct / val_total

    return val_epoch_loss, val_epoch_acc


# Add robust loss function
class FocalLoss(nn.Module):
    """Focal Loss implementation.

    Focal Loss reduces the relative loss for well-classified examples and focuses
    more on hard, misclassified examples. This can help with class imbalance and
    outliers/misclassifications.

    Args:
        alpha: Optional weighting factor for class imbalance, can be a tensor of size [num_classes].
        gamma: Focusing parameter. Higher gamma reduces the relative loss for well-classified examples.
        reduction: 'none', 'mean', or 'sum'
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions from model (before softmax) of shape [B, C]
            targets: Ground truth class indices of shape [B]

        Returns:
            loss: Scalar loss value
        """
        log_softmax = F.log_softmax(inputs, dim=1)
        CE = F.nll_loss(log_softmax, targets, reduction="none")

        # Get the correctly predicted probability
        logpt = log_softmax.gather(1, targets.unsqueeze(1))
        logpt = logpt.squeeze(1)
        pt = torch.exp(logpt)

        # Compute the focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Apply class weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha.gather(0, targets)
                focal_weight = alpha_t * focal_weight
            else:
                focal_weight = self.alpha * focal_weight

        loss = focal_weight * CE

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


# Create a robust loss that's a combination of focal loss and cross-entropy
class RobustLoss(nn.Module):
    """Robust loss function that combines Focal Loss and Cross-Entropy.

    This loss function helps to address the decorrelation between loss and accuracy
    by being less sensitive to outliers and misclassifications.

    Args:
        alpha: Weight for balancing focal loss vs cross-entropy (0-1)
        gamma: Focusing parameter for focal loss
        class_weights: Optional weighting for classes
    """

    def __init__(self, alpha=0.5, gamma=2.0, class_weights=None):
        super(RobustLoss, self).__init__()
        self.alpha = alpha
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=gamma)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions from model (before softmax) of shape [B, C]
            targets: Ground truth class indices of shape [B]

        Returns:
            loss: Combined loss value
        """
        # Calculate both losses
        fl = self.focal_loss(inputs, targets)
        ce = self.ce_loss(inputs, targets)

        # Combine losses with weighting factor
        return self.alpha * fl + (1 - self.alpha) * ce


def nn_function(
    file_paths: List[str],
    labels: List[int],
    equalize_labels: bool,
    minibatch_size: int,
    validation_patience: int,
    checkpoint_path: str,
    use_amp: bool = False,
    skip_training: bool = False,
    monitor: str = "accuracy",
) -> Tuple[nn.Module, DataLoader, DataLoader, List[int], Dict[int, int], DataLoader]:
    """Train a neural network model on the provided data.

    Args:
        file_paths: List of image file paths.
        labels: List of labels corresponding to each image.
        equalize_labels: Whether to balance the dataset.
        minibatch_size: Batch size for training.
        validation_patience: Patience for early stopping.
        checkpoint_path: Directory to save checkpoints.
        use_amp: Whether to use mixed precision training.
        skip_training: Whether to load from checkpoint instead of training.
        monitor: Metric to monitor ('loss' or 'accuracy').

    Returns:
        Tuple of (model, train_loader, val_loader, val_labels, val_label_counts, aug_val_loader)
    """
    # If equalizing labels, balance the dataset
    if equalize_labels:
        file_paths, labels = balance_dataset(file_paths, labels)

    # Split dataset into training and validation sets
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.3, stratify=labels, random_state=42
    )

    # Create data loaders
    train_dataset = ImageDataset(train_files, train_labels)
    val_dataset = ImageDataset(val_files, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=minibatch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=minibatch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Determine number of classes and initialize model
    num_classes = len(np.unique(labels))
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    model = model.to(device)

    # Set up loss function and optimizer
    criterion = RobustLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=INITIAL_LEARNING_RATE, momentum=MOMENTUM
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LEARN_RATE_DROP_PERIOD,
        gamma=LEARN_RATE_DROP_FACTOR,
    )

    # Prepare checkpoint directory
    checkpoint_path = setup_checkpoint_dir(checkpoint_path)
    acc_checkpoint_path = os.path.join(checkpoint_path, "model_checkpoint_best_acc.pt")
    loss_checkpoint_path = os.path.join(checkpoint_path, "model_checkpoint_best.pt")

    # If skip_training, try to load model from checkpoint
    if skip_training:
        # Show which checkpoint files we're looking for
        print("Looking for checkpoint files:")
        print(f"- Accuracy-based: {acc_checkpoint_path} (exists: {os.path.exists(acc_checkpoint_path)})")
        print(f"- Loss-based: {loss_checkpoint_path} (exists: {os.path.exists(loss_checkpoint_path)})")
        
        loaded = False
        
        # Try accuracy checkpoint first if monitoring accuracy
        if monitor == "accuracy" and os.path.exists(acc_checkpoint_path):
            print(f"Attempting to load accuracy-based model")
            success, message = load_model_from_checkpoint(
                model, acc_checkpoint_path, device, strict=False, verbose=False
            )
            loaded = success
            if success:
                print(f"Successfully loaded accuracy-based model")
            
        # If accuracy checkpoint failed or we're monitoring loss, try loss checkpoint
        if not loaded and os.path.exists(loss_checkpoint_path):
            print(f"Attempting to load best loss model")
            success, message = load_model_from_checkpoint(
                model, loss_checkpoint_path, device, strict=False, verbose=False
            )
            loaded = success
            if success:
                print(f"Successfully loaded loss-based model")

        if loaded:
            print("Loaded model from checkpoint, skipping training.")
            return (
                model,
                train_loader,
                val_loader,
                val_labels,
                dict(Counter(val_labels)),
                val_loader,
            )
        else:
            print("No valid checkpoint found. Training will proceed.")

    # Set up for training
    epoch_progress = tqdm(range(MAX_EPOCHS), desc="Training Progress", position=0)
    early_stopping = EarlyStopping(
        patience=validation_patience,
        verbose=True,
        path=loss_checkpoint_path,
        progress_bar=epoch_progress,
        monitor=monitor,
    )

    # Initialize tracking variables
    best_acc = 0.0
    best_acc_epoch = 0
    best_loss = float("inf")
    best_loss_epoch = 0

    # Initialize gradient scaler for mixed precision
    try:
        scaler = torch.amp.GradScaler(device_type=device.type, enabled=use_amp)
    except TypeError:
        scaler = torch.amp.GradScaler(enabled=use_amp)

    # Training loop
    for epoch_idx in epoch_progress:
        # Train for one epoch
        epoch_loss, epoch_acc = train_model(
            model, train_loader, optimizer, criterion, scaler, use_amp
        )

        # Validate after each epoch
        val_loss, val_acc = validate_model(model, val_loader, criterion, use_amp)

        # Update tracking for best accuracy and loss
        if val_acc > best_acc:
            best_acc = val_acc
            best_acc_epoch = epoch_idx

        if val_loss < best_loss:
            best_loss = val_loss
            best_loss_epoch = epoch_idx

        # Update progress bar
        epoch_progress.set_postfix(
            {
                "epoch": epoch_idx + 1,
                "train_loss": f"{epoch_loss:.4f}",
                "train_acc": f"{epoch_acc:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_acc": f"{val_acc:.4f}",
                "best_acc": f"{best_acc:.4f}",
                "best_loss": f"{best_loss:.4f}",
            }
        )

        # Step the learning rate scheduler
        scheduler.step()

        # Check early stopping
        early_stopping(val_loss, val_acc, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Load the best model based on the monitored metric
    print(f"Loading best {'accuracy' if monitor == 'accuracy' else 'loss'} model...")

    if monitor == "accuracy" and os.path.exists(acc_checkpoint_path):
        success, message = load_model_from_checkpoint(
            model, acc_checkpoint_path, device, strict=False, verbose=False
        )
        if success:
            print(f"Loaded best accuracy model (epoch {best_acc_epoch+1}, acc={best_acc:.4f})")
        else:
            print(f"Failed to load best accuracy model: {message}")
    else:
        success, message = load_model_from_checkpoint(
            model, loss_checkpoint_path, device, strict=False, verbose=False
        )
        if success:
            print(f"Loaded best loss model (epoch {best_loss_epoch+1}, loss={best_loss:.4f})")
        else:
            print(f"Failed to load best loss model: {message}")

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
            monitor="accuracy",
        )
    )
    print("Training complete.")
