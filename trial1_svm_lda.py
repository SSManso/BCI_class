from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class SVMClassifier:
    """
    Simple SVM classifier for MI/Rest classification using CSP features.

    Parameters
    ----------
    kernel : str, optional
        Kernel type ("linear" recommended). Default is "linear".conda activate
    C : float, optional
        Regularization strength. Default is 1.0.

    Expected Input to fit()
    -----------------------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix (CSP features per trial).
    y : ndarray of shape (n_samples,)
        Labels (0/1 or -1/+1).

    Output of predict()
    -------------------
    y_pred : ndarray of shape (n_samples,)
        Predicted labels.

    """
    def __init__(self, kernel="linear", C=1.0):
        self.model = SVC(kernel=kernel, C=C)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class LDAClassifier:
    """
    Linear Discriminant Analysis classifier for MI/Rest classification.

    Expected Input to fit()
    -----------------------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)

    Output of predict()
    -------------------
    y_pred : ndarray of shape (n_samples,)
    """
    def __init__(self):
        self.model = LinearDiscriminantAnalysis()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
    
# ---------------------------------------------------------- 
# Simple test code
# ---------------------------------------------------------- 
if __name__ == "__main__":
    import numpy as np
    from sklearn.metrics import accuracy_score, confusion_matrix

    # ----------------------------------------------------------
    # 1. Generate simple fake data for sanity check
    # ----------------------------------------------------------
    X = np.random.randn(20, 4)
    y = np.random.choice([0, 1], size=20)

    print("X shape:", X.shape)  # (20, 4)
    print("y shape:", y.shape)  # (20,)

    # ----------------------------------------------------------
    # 2. Test SVM classifier
    # ----------------------------------------------------------
    svm = SVMClassifier(kernel="linear")
    svm.fit(X, y)
    y_pred_svm = svm.predict(X)

    print("\nSVM predictions:", y_pred_svm)
    print("SVM accuracy:", accuracy_score(y, y_pred_svm))
    print("SVM confusion matrix:\n", confusion_matrix(y, y_pred_svm))

    # ----------------------------------------------------------
    # 3. Test LDA classifier
    # ----------------------------------------------------------
    lda = LDAClassifier()
    lda.fit(X, y)
    y_pred_lda = lda.predict(X)

    print("\nLDA predictions:", y_pred_lda)
    print("LDA accuracy:", accuracy_score(y, y_pred_lda))
    print("LDA confusion matrix:\n", confusion_matrix(y, y_pred_lda))
