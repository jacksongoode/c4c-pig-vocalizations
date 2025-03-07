from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nn_function import RobustLoss  # Import our custom robust loss
from utils import get_device

# Use the utility function to get the device
device = get_device()

# Fine-tuning hyperparameters (matching MATLAB NN_prep_for_classify.m)
FINE_TUNE_MAX_EPOCHS = 1
FINE_TUNE_MINIBATCH_SIZE = 32
FINE_TUNE_INITIAL_LEARNING_RATE = 1e-6


def nn_prep_for_classify(
    model: nn.Module, val_loader: DataLoader, train_loader: DataLoader
) -> nn.Module:
    """Fine-tune the given model for one epoch with a low learning rate using the training and validation datasets.

    This function performs a final fine-tuning step on the model to improve its classification performance.
    It uses a very low learning rate to make small adjustments to the model weights.

    Args:
        model: PyTorch model to fine-tune.
        val_loader: Validation DataLoader for evaluation during fine-tuning.
        train_loader: Training DataLoader for fine-tuning.

    Returns:
        model: The fine-tuned PyTorch model.
    """

    # Setup optimizer for fine-tuning with very low learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=FINE_TUNE_INITIAL_LEARNING_RATE)

    # Use our robust loss function for fine-tuning too
    criterion = RobustLoss()

    # Determine validation frequency: floor(number of training iterations per epoch)
    val_frequency = max(1, len(train_loader.dataset) // FINE_TUNE_MINIBATCH_SIZE)

    model.train()
    iteration = 0
    running_loss = 0.0

    # Create progress bar for fine-tuning - use position to prevent overlapping
    progress_bar = tqdm(train_loader, desc="Fine-tuning", leave=True, position=0)

    for inputs, target_labels in progress_bar:
        iteration += 1
        inputs, target_labels = inputs.to(device), target_labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, target_labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Calculate batch accuracy for progress bar
        _, predicted = torch.max(outputs.data, 1)
        batch_acc = (predicted == target_labels).sum().item() / target_labels.size(0)

        # Update progress bar
        progress_bar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}", "iter": iteration}
        )

        # Mid-epoch validation
        if iteration % val_frequency == 0:
            val_metrics = validate_model(model, val_loader, criterion)
            progress_bar.write(
                f"[Iteration {iteration}] Fine-tuning - "
                f"Mid-epoch Val Loss: {val_metrics[0]:.4f}, Val Acc: {val_metrics[1]:.4f}"
            )
            model.train()  # Switch back to training mode

    # End of epoch: Evaluate on validation set
    val_loss, val_acc = validate_model(model, val_loader, criterion)
    progress_bar.write(
        f"Fine-tuning - Final Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )

    return model


def validate_model(
    model: nn.Module, val_loader: DataLoader, criterion: nn.Module
) -> Tuple[float, float]:
    """Evaluate the model on the validation set.

    Args:
        model: PyTorch model to evaluate.
        val_loader: Validation DataLoader.
        criterion: Loss function.

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


if __name__ == "__main__":
    # Example usage would require an already created model and datasets
    print(
        "nn_prep_for_classify module loaded. Replace with actual model and datasets for testing."
    )
