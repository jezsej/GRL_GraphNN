from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score
import numpy as np


def compute_classification_metrics(y_true, y_pred, y_score=None):
    """
    Computes accuracy, sensitivity, specificity, F1-score, and ROC-AUC.
    Args:
        y_true (np.ndarray): Ground truth labels
        y_pred (np.ndarray): Predicted labels
        y_score (np.ndarray or None): Prediction scores/probabilities (for AUC)
    Returns:
        dict: Dictionary of computed metrics
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['sensitivity'] = tp / (tp + fn + 1e-6)
    metrics['specificity'] = tn / (tn + fp + 1e-6)

    if y_score is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_score)
        except:
            metrics['auc'] = float('nan')
    else:
        metrics['auc'] = None

    return metrics
