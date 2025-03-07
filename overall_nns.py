import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from metrics import f1_metrics
from nn_function import nn_function
from nn_prep_for_classify import nn_prep_for_classify
from utils import get_device

# Use the utility function to get the device
device = get_device()

# Helper function to classify a dataset using the PyTorch model


def classify_dataset(model, dataloader):
    """Return predicted labels for the dataset."""
    model.eval()  # Set the model to evaluation mode
    all_preds = []

    # Add progress bar for prediction - use position to prevent overlapping
    progress_bar = tqdm(dataloader, desc="Predicting", leave=False, position=1)

    with torch.no_grad():
        for inputs, _ in progress_bar:
            inputs = inputs.to(device)

            # Handle different PyTorch versions for autocast
            try:
                # Try with device_type parameter (newer PyTorch versions)
                with torch.amp.autocast(
                    device_type=device.type, enabled=(device.type != "cpu")
                ):
                    outputs = model(inputs)
            except TypeError:
                # Fall back to older PyTorch versions or CPU
                if device.type == "cuda":
                    # For older PyTorch with CUDA
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                else:
                    # For CPU
                    outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())

    return np.array(all_preds)


# Helper function to compute overall accuracy


def compute_accuracy(pred_labels, true_labels):
    import torch

    if torch.is_tensor(true_labels):
        true_labels = true_labels.cpu().numpy().astype(int)
    else:
        true_labels = np.array(true_labels, dtype=int)
    return np.sum(pred_labels == true_labels) / len(true_labels)


# Added helper function to run training, fine-tuning and evaluation for given label type


def run_evaluation(
    files,
    numeric_labels,
    equalize,
    minibatch_size,
    validation_patience,
    checkpoint_path,
    skip_training=False,
):
    model, train_loader, val_loader, lbls, lbl_counts, _ = nn_function(
        files,
        numeric_labels,
        equalize_labels=equalize,
        minibatch_size=minibatch_size,
        validation_patience=validation_patience,
        checkpoint_path=checkpoint_path,
        use_amp=(device.type == "cuda"),
        skip_training=skip_training,
        monitor="accuracy",  # Use accuracy-based model selection
    )

    # Fine-tune for classification
    model = nn_prep_for_classify(model, val_loader, train_loader)

    # Get predictions
    preds = classify_dataset(model, val_loader)

    # Compute overall accuracy using true labels extracted from the validation dataset
    true_labels = np.array(val_loader.dataset.labels, dtype=int)
    acc = compute_accuracy(preds, true_labels)

    # Compute confusion matrix and metrics
    conf = confusion_matrix(true_labels, preds)
    p, r, f1, wp, wr, wf1 = f1_metrics(conf, list(lbl_counts.values()))

    metrics = {
        "accuracy": acc,
        "weighted_precision": wp,
        "weighted_recall": wr,
        "weighted_f1": wf1,
        "preds": preds,
        "labels": true_labels,
        "confusion": conf,
    }
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="Skip training and load the last best checkpoint",
    )
    parser.add_argument(
        "--label_type",
        type=str,
        default="both",
        choices=["both", "valence", "context"],
        help="Which label type to run: both, valence, or context",
    )
    args = parser.parse_args()

    # Read data from Excel, adjust file name and columns as needed
    # Assuming the Excel file is named 'SoundwelDatasetKey.xlsx' and is in the current directory
    excel_file = "SoundwelDatasetKey.xlsx"
    data = pd.read_excel(excel_file)

    # Extract columns (adjust column names/indexes based on actual data)
    # For this example, we assume the Excel has columns: 'File', 'Valence', 'Context', 'Site'
    Files = data["Spectrogram Filename"].tolist()

    # Append directory to Files
    Files = [os.path.join("soundwel", f) for f in Files]

    Valence = data["Valence"].tolist()
    Context = data["Context"].tolist()  # Uncommented to handle context classification
    Site = data["Recording Team"].tolist()

    # Convert labels to numeric categories
    # Create mappings for valence and context
    valence_categories = {cat: idx for idx, cat in enumerate(sorted(set(Valence)))}
    context_categories = {
        cat: idx for idx, cat in enumerate(sorted(set(Context)))
    }  # Added for context
    site_labels = sorted(set(Site))

    Valence_numeric = [valence_categories[x] for x in Valence]
    Context_numeric = [context_categories[x] for x in Context]  # Added for context

    # Define checkpoint directory bases
    checkpoint_base_val = os.path.join("checkpoints", "valence")
    checkpoint_base_con = os.path.join("checkpoints", "context")
    os.makedirs(checkpoint_base_val, exist_ok=True)
    os.makedirs(checkpoint_base_con, exist_ok=True)

    # Dictionaries to store metrics
    overall_metrics_val = {}
    overall_metrics_con = {}  # Added for context
    site_accuracy_val = defaultdict(dict)
    site_accuracy_con = defaultdict(dict)  # Added for context

    # Create a progress bar for iterations - use position to prevent overlapping
    iteration_number = 1  # We only need one iteration with our improved approach
    iterations = range(1, iteration_number + 1)

    if args.label_type == "both":
        desc = "Training valence & context models"
    elif args.label_type == "valence":
        desc = "Training valence models"
    else:
        desc = "Training context models"

    iteration_progress = tqdm(iterations, desc=desc, position=0)

    # Loop over iterations (reduced to just 1 iteration with our improved approach)
    for i in iteration_progress:
        # Define checkpoint directory for this iteration
        cp_val = os.path.join(checkpoint_base_val, f"{i}")
        cp_con = os.path.join(checkpoint_base_con, f"{i}")
        os.makedirs(cp_val, exist_ok=True)
        os.makedirs(cp_con, exist_ok=True)

        val_metrics_dict = None
        con_metrics_dict = None

        if args.label_type in ["both", "valence"]:
            # Run evaluation for valence
            iteration_progress.set_postfix({"model": "valence"})
            val_metrics_dict = run_evaluation(
                Files,
                Valence_numeric,
                True,
                32,
                5,
                cp_val,
                skip_training=args.skip_training,
            )
            iteration_progress.write(
                f"Iteration {i} - Valence Accuracy: {val_metrics_dict['accuracy']:.2f}"
            )
            overall_metrics_val[i] = val_metrics_dict
            for site in site_labels:
                site_accuracy_val[site][i] = val_metrics_dict["accuracy"]

        if args.label_type in ["both", "context"]:
            # Run evaluation for context
            iteration_progress.set_postfix({"model": "context"})
            con_metrics_dict = run_evaluation(
                Files,
                Context_numeric,
                False,
                32,
                5,
                cp_con,
                skip_training=args.skip_training,
            )
            iteration_progress.write(
                f"Iteration {i} - Context Accuracy: {con_metrics_dict['accuracy']:.2f}"
            )
            overall_metrics_con[i] = con_metrics_dict
            for site in site_labels:
                site_accuracy_con[site][i] = con_metrics_dict["accuracy"]

        # Update progress bar with current metrics
        if args.label_type == "both":
            iteration_progress.set_postfix(
                {
                    "valence_acc": f"{val_metrics_dict['accuracy']:.2f}",
                    "context_acc": f"{con_metrics_dict['accuracy']:.2f}",
                }
            )
        elif args.label_type == "valence" and val_metrics_dict:
            iteration_progress.set_postfix(
                {"accuracy": f"{val_metrics_dict['accuracy']:.2f}"}
            )
        elif args.label_type == "context" and con_metrics_dict:
            iteration_progress.set_postfix(
                {"accuracy": f"{con_metrics_dict['accuracy']:.2f}"}
            )

    # Save metrics to file or print summary
    print("\nTraining Complete! Summary of Results:")

    if args.label_type in ["both", "valence"]:
        val_accs = [metrics["accuracy"] for metrics in overall_metrics_val.values()]
        iteration_progress.write(
            f"Valence model - Avg Accuracy: {np.mean(val_accs):.4f}, Best: {np.max(val_accs):.4f}"
        )

    if args.label_type in ["both", "context"]:
        con_accs = [metrics["accuracy"] for metrics in overall_metrics_con.values()]
        iteration_progress.write(
            f"Context model - Avg Accuracy: {np.mean(con_accs):.4f}, Best: {np.max(con_accs):.4f}"
        )

    # New block to run site_validation_imds from the nn_function overall
    try:
        from site_validation_imds import site_validation_imds

        # Define the directory containing the soundwel images
        image_dir = "soundwel"
        # List image files in the directory with extensions png, jpg, jpeg
        image_files = [
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        # Create dummy labels (e.g., all zeros)
        dummy_labels = [0] * len(image_files)

        iteration_progress.write("Running site_validation_imds on image files...")
        dataset_site = site_validation_imds(
            image_files, dummy_labels, base_dir=image_dir
        )
        # Get the first batch to check
        images, labels = next(iter(dataset_site))
        iteration_progress.write(
            f"Site validation batch images shape: {images.shape}, labels: {labels}"
        )
    except Exception as e:
        iteration_progress.write(f"Error during site validation: {e}")
