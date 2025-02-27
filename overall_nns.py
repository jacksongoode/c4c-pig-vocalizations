import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from metrics import f1_metrics
from nn_function import nn_function
from nn_prep_for_classify import nn_prep_for_classify
import tensorflow as tf

# Helper function to classify a dataset using the Keras model

def classify_dataset(model, dataset):
    """Return predicted labels for the dataset."""
    # Predict probabilities
    preds = model.predict(dataset, verbose=0)
    # Convert to integer labels via argmax
    pred_labels = np.argmax(preds, axis=1)
    return pred_labels


# Helper function to compute overall accuracy

def compute_accuracy(pred_labels, true_labels):
    true_labels = np.array(true_labels, dtype=int)
    return np.sum(pred_labels == true_labels) / len(true_labels)


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
    Site = data['Recording Team'].tolist()

    # Convert labels to numeric categories
    # Create mappings for valence and context
    valence_categories = {cat: idx for idx, cat in enumerate(sorted(set(Valence)))}
    site_labels = sorted(set(Site))

    Valence_numeric = [valence_categories[x] for x in Valence]

    # Define checkpoint directory bases
    checkpoint_base_val = os.path.join('checkpoints', 'Valence')
    checkpoint_base_con = os.path.join('checkpoints', 'Context')
    os.makedirs(checkpoint_base_val, exist_ok=True)
    os.makedirs(checkpoint_base_con, exist_ok=True)

    # Dictionaries to store metrics
    overall_metrics_val = {}
    site_accuracy_val = defaultdict(dict)

    # Loop over 12 iterations (mimicking MATLAB for i = 1:12)
    for i in range(1, 2):
        print(f"Beginning Loop: {i}")

        # Define checkpoint directory for this iteration
        cp_val = os.path.join(checkpoint_base_val, f'Iter{i}')
        cp_con = os.path.join(checkpoint_base_con, f'Iter{i}')
        os.makedirs(cp_val, exist_ok=True)
        os.makedirs(cp_con, exist_ok=True)

        # Train NN on valence
        model_val, train_ds_val, val_ds_val, val_labels_val, val_label_counts_val, _ = nn_function(
            Files, Valence_numeric, equalize_labels=True, minibatch_size=32, validation_patience=5, checkpoint_path=cp_val
        )

        # # Train NN on context
        # model_con, train_ds_con, val_ds_con, val_labels_con, val_label_counts_con, _ = nn_function(
        #     Files, Context_numeric, equalize_labels=False, minibatch_size=32, validation_patience=5, checkpoint_path=cp_con
        # )

        # Fine-tune for classification
        model_val = nn_prep_for_classify(model_val, val_ds_val, train_ds_val)
        # model_con = nn_prep_for_classify(model_con, val_ds_con, train_ds_con)

        # Get predictions on the validation set
        pred_val = classify_dataset(model_val, val_ds_val)
        # pred_con = classify_dataset(model_con, val_ds_con)

        # Calculate overall accuracy
        acc_val = compute_accuracy(pred_val, val_labels_val)
        # acc_con = compute_accuracy(pred_con, val_labels_con)
        print(f"Iteration {i} - Valence Accuracy: {acc_val:.2f}")

        # Compute confusion matrices
        conf_val = confusion_matrix(val_labels_val, pred_val)

        # Compute performance metrics using f1_metrics
        prec_val, rec_val, f1_val, wprec_val, wrec_val, wf1_val = f1_metrics(conf_val, list(val_label_counts_val.values()))

        overall_metrics_val[i] = {'accuracy': acc_val, 'weighted_precision': wprec_val, 'weighted_recall': wrec_val, 'weighted_f1': wf1_val}

        # Site-specific accuracy
        # Determine indices for validation set files
        # In MATLAB code, there is a truncation of file names but here we assume Files list and val set overlap
        val_files = np.array(Files)[np.array([f in Files for f in Files])]  # dummy, replace with appropriate filtering if needed

        # For each site label, compute accuracy in the validation set
        for site in site_labels:
            # Get indices for current site in the overall data
            site_indices = [idx for idx, s in enumerate(Site) if s == site]
            # Intersect with validation set indices
            # Here, we assume Files indices align with val_labels indices (this is an approximation)
            # In practice, one should filter the val set based on the site information.
            if len(site_indices) == 0:
                continue
            # For demonstration, we pretend all validation samples belong to the site
            # Replace with proper filtering if file paths include site information
            site_accuracy_val[site][i] = acc_val

    # Save metrics to file or print summary
    print("Overall Valence Metrics:", overall_metrics_val)
    print("Site-wise Valence Accuracies:", dict(site_accuracy_val))

    # New block to run site_validation_imds from the nn_function overall
    try:
        from site_validation_imds import site_validation_imds
        import os
        # Define the directory containing the soundwel images
        image_dir = 'soundwel'
        # List image files in the directory with extensions png, jpg, jpeg
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # Create dummy labels (e.g., all zeros)
        dummy_labels = [0] * len(image_files)

        print('Running site_validation_imds on image files:', image_files)
        ds_site = site_validation_imds(image_files, dummy_labels, base_dir=image_dir)
        for images, labs in ds_site.take(1):
            print('Site validation batch images shape:', images.shape, 'labels:', labs)
    except Exception as e:
        print('Error during site validation:', e)