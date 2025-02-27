import torch
from torch import nn

# Check if MPS is available for Apple Silicon
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


def nn_prep_for_classify(model, val_loader, train_loader):
    """Fine-tune the given model for one epoch with a low learning rate using the training and validation datasets.

    Parameters:
        model: PyTorch model.
        val_loader: Validation DataLoader.
        train_loader: Training DataLoader.

    Returns:
        model: The fine-tuned PyTorch model.
    """
    # Set a very low learning rate for fine-tuning
    low_lr = 1e-6

    # Setup optimizer for fine-tuning
    optimizer = torch.optim.SGD(model.parameters(), lr=low_lr)
    criterion = nn.CrossEntropyLoss()

    # Fine-tune for 1 epoch
    model.train()
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

    # Evaluate on validation set
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

    print(f'Fine-tuning - Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

    return model


if __name__ == '__main__':
    # Example usage would require an already created model and datasets
    print('nn_prep_for_classify module loaded. Replace with actual model and datasets for testing.')