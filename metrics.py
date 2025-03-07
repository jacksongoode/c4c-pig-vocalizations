import numpy as np


def precision(M):
    """Calculate the precision vector from a confusion matrix M.
    Precision is defined as diag(M) / sum(M, axis=1).
    Parameters:
        M (np.ndarray): Confusion matrix.
    Returns:
        np.ndarray: The per-class precision.
    """
    M = np.array(M, dtype=float)
    # Avoid division by zero by adding eps
    eps = np.finfo(float).eps
    p = np.diag(M) / (np.sum(M, axis=1) + eps)
    return p


def recall(M):
    """Calculate the recall vector from a confusion matrix M.
    Recall is defined as diag(M) / sum(M, axis=0).
    Parameters:
        M (np.ndarray): Confusion matrix.
    Returns:
        np.ndarray: The per-class recall.
    """
    M = np.array(M, dtype=float)
    eps = np.finfo(float).eps
    r = np.diag(M) / (np.sum(M, axis=0) + eps)
    return r


def f1_score(p, r):
    """Calculate the F1 score given precision and recall vectors.
    The F1 score is defined as 2*(p*r)/(p+r). If (p+r)==0, F1 is set to 0.
    Parameters:
        p (np.ndarray): Precision vector.
        r (np.ndarray): Recall vector.
    Returns:
        np.ndarray: The per-class F1 score.
    """
    eps = np.finfo(float).eps
    f = np.where((p + r) > 0, 2 * p * r / (p + r + eps), 0)
    return f


def f1_metrics(M, label_counts):
    """Compute per-class precision, recall, F1 score and weighted metrics.

    Parameters:
        M (np.ndarray): Confusion matrix.
        label_counts (array-like): Counts for each label (should sum to the total).

    Returns:
        tuple: (precision_vector, recall_vector, f1_vector, weighted_precision, weighted_recall, weighted_f1)
    """
    label_counts = np.array(label_counts, dtype=float)
    weights = label_counts / (np.sum(label_counts) + np.finfo(float).eps)
    p = precision(M)
    r = recall(M)
    f = f1_score(p, r)
    weighted_p = np.sum(p * weights)
    weighted_r = np.sum(r * weights)
    weighted_f = np.sum(f * weights)
    return p, r, f, weighted_p, weighted_r, weighted_f
