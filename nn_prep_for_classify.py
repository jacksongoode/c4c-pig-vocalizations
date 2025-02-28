import torch
from torch import nn

# Check if MPS is available for Apple Silicon
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

# Fine-tuning hyperparameters (matching MATLAB NN_prep_for_classify.m)
FINE_TUNE_MAX_EPOCHS = 1
FINE_TUNE_MINIBATCH_SIZE = 32
FINE_TUNE_INITIAL_LEARNING_RATE = 1e-6


def nn_prep_for_classify(model, val_loader, train_loader):
    """Fine-tune the given model for one epoch with a low learning rate using the training and validation datasets.

    Parameters:
        model: PyTorch model.
        val_loader: Validation DataLoader.
        train_loader: Training DataLoader.

    Returns:
        model: The fine-tuned PyTorch model.
    """

    # Setup optimizer for fine-tuning with very low learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=FINE_TUNE_INITIAL_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Determine validation frequency: floor(number of training iterations per epoch)
    val_frequency = max(1, len(train_loader.dataset) // FINE_TUNE_MINIBATCH_SIZE)

    model.train()
    iteration = 0
    running_loss = 0.0

    for inputs, target_labels in train_loader:
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

        # Mid-epoch validation
        if iteration % val_frequency == 0:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_outputs = model(val_inputs)
                    v_loss = criterion(val_outputs, val_labels)
                    val_loss += v_loss.item() * val_inputs.size(0)
                    _, val_predicted = torch.max(val_outputs.data, 1)
                    val_total += val_labels.size(0)
                    val_correct += (val_predicted == val_labels).sum().item()
            current_val_loss = val_loss / val_total
            current_val_acc = val_correct / val_total
            print(f"[Iteration {iteration}] Fine-tuning - Mid-epoch Val Loss: {current_val_loss:.4f}, Val Acc: {current_val_acc:.4f}")
            model.train()  # Switch back to training mode

    # End of epoch: Evaluate on validation set
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

    print(f'Fine-tuning - Final Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

    return model


if __name__ == '__main__':
    # Example usage would require an already created model and datasets
    print('nn_prep_for_classify module loaded. Replace with actual model and datasets for testing.')