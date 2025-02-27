import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from metrics import f1_metrics
from nn_function import nn_function
from nn_prep_for_classify import nn_prep_for_classify

# Check if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Helper function to classify a dataset using the PyTorch model

def classify_dataset(model, dataloader):
    """Return predicted labels for the dataset."""
    model.eval()  # Set the model to evaluation mode
    all_preds = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            # Use mixed precision if on CUDA
            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())

    return np.array(all_preds)


# Helper function to compute overall accuracy

def compute_accuracy(pred_labels, true_labels):
    true_labels = np.array(true_labels, dtype=int)
    return np.sum(pred_labels == true_labels) / len(true_labels)


# Added helper function to run training, fine-tuning and evaluation for given label type

def run_evaluation(files, numeric_labels, equalize, minibatch_size, validation_patience, checkpoint_path):
    # Train NN with mixed precision support if using CUDA
    model, train_loader, val_loader, lbls, lbl_counts, _ = nn_function(
        files, numeric_labels, equalize_labels=equalize, minibatch_size=minibatch_size,
        validation_patience=validation_patience, checkpoint_path=checkpoint_path, use_amp=(device.type=='cuda')
    )

    # Fine-tune for classification
    model = nn_prep_for_classify(model, val_loader, train_loader)

    # Get predictions
    preds = classify_dataset(model, val_loader)
    # Compute overall accuracy
    acc = compute_accuracy(preds, lbls)
    # Compute confusion matrix and metrics
    conf = confusion_matrix(lbls, preds)
    p, r, f1, wp, wr, wf1 = f1_metrics(conf, list(lbl_counts.values()))

    metrics = {
        'accuracy': acc,
        'weighted_precision': wp,
        'weighted_recall': wr,
        'weighted_f1': wf1,
        'preds': preds,
        'labels': lbls,
        'confusion': conf
    }
    return metrics


if __name__ == '__main__':
    # Read data from Excel, adjust file name and columns as needed
    # Assuming the Excel file is named 'SoundwelDatasetKey.xlsx' and is in the current directory
    excel_file = 'SoundwelDatasetKey.xlsx'
    data = pd.read_excel(excel_file)

    # Extract columns (adjust column names/indexes based on actual data)
    # For this example, we assume the Excel has columns: 'File', 'Valence', 'Context', 'Site'
    Files = data['Spectrogram Filename'].tolist()

    # Append directory to Files
    Files = [os.path.join('soundwel', f) for f in Files]

    Valence = data['Valence'].tolist()
    Context = data['Context'].tolist()  # Uncommented to handle context classification
    Site = data['Recording Team'].tolist()

    # Convert labels to numeric categories
    # Create mappings for valence and context
    valence_categories = {cat: idx for idx, cat in enumerate(sorted(set(Valence)))}
    context_categories = {cat: idx for idx, cat in enumerate(sorted(set(Context)))}  # Added for context
    site_labels = sorted(set(Site))

    Valence_numeric = [valence_categories[x] for x in Valence]
    Context_numeric = [context_categories[x] for x in Context]  # Added for context

    # Define checkpoint directory bases
    checkpoint_base_val = os.path.join('checkpoints', 'Valence')
    checkpoint_base_con = os.path.join('checkpoints', 'Context')
    os.makedirs(checkpoint_base_val, exist_ok=True)
    os.makedirs(checkpoint_base_con, exist_ok=True)

    # Dictionaries to store metrics
    overall_metrics_val = {}
    overall_metrics_con = {}  # Added for context
    site_accuracy_val = defaultdict(dict)
    site_accuracy_con = defaultdict(dict)  # Added for context

    # Loop over 12 iterations (mimicking MATLAB for i = 1:12)
    for i in range(1, 13):
        print(f"Beginning Loop: {i}")

        # Define checkpoint directory for this iteration
        cp_val = os.path.join(checkpoint_base_val, f'Iter{i}')
        cp_con = os.path.join(checkpoint_base_con, f'Iter{i}')
        os.makedirs(cp_val, exist_ok=True)
        os.makedirs(cp_con, exist_ok=True)

        # Run evaluation for valence
        val_metrics_dict = run_evaluation(Files, Valence_numeric, True, 32, 5, cp_val)
        # Run evaluation for context
        con_metrics_dict = run_evaluation(Files, Context_numeric, False, 32, 5, cp_con)

        print(f"Iteration {i} - Valence Accuracy: {val_metrics_dict['accuracy']:.2f}")
        print(f"Iteration {i} - Context Accuracy: {con_metrics_dict['accuracy']:.2f}")

        overall_metrics_val[i] = val_metrics_dict
        overall_metrics_con[i] = con_metrics_dict

        # Site-specific accuracy (Note: This section is a placeholder. In the MATLAB version, filenames are truncated for filtering.
        # Here, proper filtering logic should be implemented based on your dataset structure. For now, we assign overall accuracy to each site.)
        for site in site_labels:
            site_accuracy_val[site][i] = val_metrics_dict['accuracy']
            site_accuracy_con[site][i] = con_metrics_dict['accuracy']

    # Save metrics to file or print summary
    print("Overall Valence Metrics:", overall_metrics_val)
    print("Overall Context Metrics:", overall_metrics_con)
    print("Site-wise Valence Accuracies:", dict(site_accuracy_val))
    print("Site-wise Context Accuracies:", dict(site_accuracy_con))

    # New block to run site_validation_imds from the nn_function overall
    try:
        from site_validation_imds import site_validation_imds
        # Define the directory containing the soundwel images
        image_dir = 'soundwel'
        # List image files in the directory with extensions png, jpg, jpeg
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # Create dummy labels (e.g., all zeros)
        dummy_labels = [0] * len(image_files)

        print('Running site_validation_imds on image files:', image_files)
        dataset_site = site_validation_imds(image_files, dummy_labels, base_dir=image_dir)
        # Get the first batch to check
        images, labels = next(iter(dataset_site))
        print('Site validation batch images shape:', images.shape, 'labels:', labels)
    except Exception as e:
        print('Error during site validation:', e)