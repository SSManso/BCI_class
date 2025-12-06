## File to define the function(s) used to control the decoder section of the project

## Imports

# From this project (constants)
from utils.constants import MODEL_SVM_LINEAR, MODEL_SVM_RBF, MODEL_LDA
from utils.constants import RANDOM_SEED
from utils.constants import METRIC_ACC

# From this project (functions)
# - None

# From sys library
import logging

# From external libraries
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

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
    acc = accuracy_score(labels, pred_test_labels)
    
    # Create metric
    metric = {
        METRIC_ACC: acc
    }

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
    all_acc = []
    for metric in list_metrics:
        this_acc = metric[METRIC_ACC]
        all_acc.append(this_acc)
    avg_acc = np.mean(all_acc)

    # Create new metric
    avg_metric = {
        METRIC_ACC: avg_acc
    }

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