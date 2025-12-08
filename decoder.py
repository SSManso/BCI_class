## File to define the function(s) used to control the decoder section of the project

## Imports

# From this project (constants)
from utils.constants import MODEL_SVM_LINEAR, MODEL_SVM_RBF, MODEL_LDA, MODEL_MDM_RIEM, MODEL_MDM_EUC
from utils.variable_constants import RANDOM_SEED
from utils.constants import METRIC_ACC, METRIC_PREC, METRIC_REC, METRIC_F1, METRIC_BAL_ACC, METRIC_CONFMAT, METRIC_TPR, METRIC_TNR, METRIC_FPR, METRIC_FNR, METRIC_TP, METRIC_TN, METRIC_FP, METRIC_FN

# From this project (functions)
# - None

# From sys library
import logging

# From external libraries
import numpy as np
from pyriemann.classification import MDM
from sklearn.base import ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt

## Set up logger
logger = logging.getLogger(__name__)


## Functions

# TODO
def create_model(model_option: str) -> ClassifierMixin:
    """
    TODO explain

    Parameters:
        model_option (str): TODO explain

    Returns:
        ClassifierMixin: TODO explain
    """

    # Create model
    if model_option == MODEL_SVM_LINEAR:
        model = SVC(kernel='linear', random_state=RANDOM_SEED)
    elif model_option == MODEL_SVM_RBF:
        model = SVC(kernel='rbf', random_state=RANDOM_SEED)
    elif model_option == MODEL_LDA:
        model = LinearDiscriminantAnalysis()
    elif model_option == MODEL_MDM_RIEM:
        model = MDM(metric='riemann')  
    elif model_option == MODEL_MDM_EUC:
        model = MDM(metric='euclid') 
    else:
        logger.error(f'Unexpected model option')
        raise ValueError('Unexpected model option')

    return model

# TODO
def train_model(model: ClassifierMixin, features: np.ndarray, labels: np.ndarray) -> tuple[ClassifierMixin, dict[str, float]]:
    """
    TODO explain

    Parameters:
        model (ClassifierMixin): TODO explain
        features (np.ndarray): TODO explain
        label (np.ndarray): TODO explain

    Returns:
        tuple[ClassifierMixin, dict[str, float]]: TODO explain
    """

    # Train model
    model = model.fit(features, labels)

    # See train performance
    pred_train_labels = model.predict(features)
    acc = accuracy_score(labels, pred_train_labels)
    
    # Create metric
    metric = {
        METRIC_ACC: acc
    }

    return model, metric

# TODO
def test_model(model: ClassifierMixin, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """
    TODO explain

    Parameters:
        model (ClassifierMixin): TODO explain
        features (np.ndarray): TODO explain
        label (np.ndarray): TODO explain

    Returns:
        dict[str, float]: TODO explain
    """

    # See test performance
    pred_test_labels = model.predict(features)
    metric = compute_metrics(labels, pred_test_labels)

    return metric

# TODO
def average_list_metrics(list_metrics: list[dict[str, float]]) -> dict[str, float]:
    """
    TODO explain

    Parameters:
        list_metrics list[dict[str, float]]: TODO explain

    Returns:
        dict[str, float]: TODO explain
    """

    # Get average metric values
    if len(list_metrics) == 0:
        return {}

    keys = list_metrics[0].keys()
    avg_metric = {}

    for k in keys:
        avg_metric[k] = float(np.mean([m[k] for m in list_metrics]))

    return avg_metric

# TODO
def max_list_metrics(list_metrics: list[dict[str, float]]):
    """
    TODO explain

    Parameters:
        list_metrics list[dict[str, float]]: TODO explain

    Returns:
        dict[str, float]: TODO explain
    """

    # Get max metric
    max_acc = -np.inf
    max_metric = None
    for metric in list_metrics:
        this_acc = metric[METRIC_ACC]
        if this_acc > max_acc:
            max_acc = this_acc
            max_metric = metric

    return max_metric

def log_metrics(logger, name: str, metric_dict: dict, decimals: int = 3):
    """Pretty-print all scalar metrics in a consistent way."""
    metrics_str = ', '.join(
        f'{k}={v:.{decimals}f}' for k, v in metric_dict.items()
    )
    logger.info(f'{name} metrics: {metrics_str}')

def compute_metrics(true, pred) -> dict[str, float]:
    # Basic metrics
    metrics = {
        METRIC_ACC: accuracy_score(true, pred),
        METRIC_PREC: precision_score(true, pred, zero_division=0),
        METRIC_REC: recall_score(true, pred, zero_division=0),
        METRIC_F1: f1_score(true, pred, zero_division=0),
        METRIC_BAL_ACC: balanced_accuracy_score(true, pred),
    }

    # Confusion-matrix-derived
    cm = confusion_matrix(true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics.update({
        METRIC_TP: float(tp),
        METRIC_TN: float(tn),
        METRIC_FP: float(fp),
        METRIC_FN: float(fn),
        METRIC_TPR: tpr,
        METRIC_TNR: tnr,
        METRIC_FPR: fpr,
        METRIC_FNR: fnr,
    })

    return metrics

def compute_epoch_level_accuracy(model, features, labels, epochs) -> float:
    """
    Compute epoch-level accuracy by majority voting over window predictions.

    Parameters:
        model: trained classifier with a .predict() method
        features (np.ndarray): shape (n_windows, n_features)
        labels (array-like): per-window true labels
        epochs (array-like): per-window epoch IDs (same length as labels)

    Returns:
        float: epoch-level accuracy in [0, 1]
    """
    labels = np.array(labels)
    epochs = np.array(epochs)

    # Per-window predictions
    y_pred = model.predict(features)

    epoch_true = []
    epoch_pred = []

    for eid in np.unique(epochs):
        idx = (epochs == eid)

        # All windows in this epoch share the same true label
        true_label = labels[idx][0]

        # Majority vote over predicted labels in this epoch
        preds_epoch = y_pred[idx]
        vals, counts = np.unique(preds_epoch, return_counts=True)
        majority_label = vals[np.argmax(counts)]

        epoch_true.append(true_label)
        epoch_pred.append(majority_label)

    epoch_true = np.array(epoch_true)
    epoch_pred = np.array(epoch_pred)

    return float((epoch_true == epoch_pred).mean())


def compute_epoch_level_predictions(
    model,
    features: np.ndarray,
    labels,
    epochs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Majority-vote predictions per epoch.

    Returns:
        epoch_true: shape (n_epochs,)
        epoch_pred: shape (n_epochs,)
    """
    labels = np.array(labels)
    epochs = np.array(epochs)

    # Per-window predictions
    y_pred = model.predict(features)

    epoch_true = []
    epoch_pred = []

    for eid in np.unique(epochs):
        idx = (epochs == eid)

        # All windows in this epoch share the same true label
        true_label = labels[idx][0]

        # Majority vote over predicted labels in this epoch
        preds_epoch = y_pred[idx]
        vals, counts = np.unique(preds_epoch, return_counts=True)
        majority_label = vals[np.argmax(counts)]

        epoch_true.append(true_label)
        epoch_pred.append(majority_label)

    return np.array(epoch_true), np.array(epoch_pred)


def compute_epoch_level_metrics(
    model,
    features: np.ndarray,
    labels,
    epochs,
    prefix: str = "epoch_",
) -> dict[str, float]:
    """
    Compute metrics on epoch-level majority-vote predictions.

    prefix: added to each metric key (e.g., "epoch_acc", "epoch_tpr", ...)
    """
    epoch_true, epoch_pred = compute_epoch_level_predictions(
        model, features, labels, epochs
    )

    base_metrics = compute_metrics(epoch_true, epoch_pred)

    if prefix:
        base_metrics = {f"{prefix}{k}": v for k, v in base_metrics.items()}

    return base_metrics


def compute_epoch_level_accuracy(
    model,
    features: np.ndarray,
    labels,
    epochs,
) -> float:
    """
    """
    epoch_true, epoch_pred = compute_epoch_level_predictions(
        model, features, labels, epochs
    )
    return float(accuracy_score(epoch_true, epoch_pred))


def plot_confusion_matrix(
    true,
    pred,
    class_names=("REST", "MI"),
    title="Confusion Matrix",
    normalize=False,
    save_path=None,
):
    """
    Plot a window-level or epoch-level confusion matrix.

    Parameters:
        true, pred: arrays of true and predicted labels
        class_names: list or tuple of class names in order [0,1]
        title: title for the plot
        normalize: whether to convert counts to percentages
        save_path: if given, saves the figure instead of plt.show()
    """
    cm = confusion_matrix(true, pred, labels=[0, 1])

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # avoid NaNs if a row is zero

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    # Ticks
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title(title) 
    # Annotate cells
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.colorbar(im, ax=ax)

    if save_path:
        plt.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()