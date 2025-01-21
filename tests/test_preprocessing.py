import pytest
import numpy as np
from regression.preprocessing import remove_outliers, normalize_features, train_test_split

class TestPreprocessing:
    """
    Test class for preprocessing functions: remove_outliers, normalize_features, and train_test_split.
    """

    def test_normalize_features(self):
        """
        Test normalize_features to ensure the result has mean 0 and std 1.
        """
        X = np.array([[1, 2], [3, 4], [5, 6]])
        normalized_X = normalize_features(X)
        
        # Test mean is close to 0
        assert np.allclose(np.mean(normalized_X, axis=0), 0, atol=1e-8), "Mean is not approximately 0."
        # Test std is close to 1
        assert np.allclose(np.std(normalized_X, axis=0), 1, atol=1e-8), "Standard deviation is not approximately 1."

    def test_remove_outliers(self):
        """
        Test remove_outliers to ensure proper outlier removal.
        """
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([10, 12, 14, 100])  # 100 is an outlier

        X_clean, y_clean = remove_outliers(X, y)

        # Assert no outlier values in y_clean
        assert len(y_clean) == 3, "Number of samples after outlier removal is incorrect."
        assert np.all(y_clean <= 14) and np.all(y_clean >= 10), "Outliers not correctly removed from y."
        # Ensure the corresponding X values are correctly filtered
        assert X_clean.shape[0] == y_clean.shape[0], "Mismatch between cleaned X and y shapes."

    def test_train_test_split(self):
        """
        Test train_test_split for correct train-test separation and proportions.
        """
        X = np.array([[i, i+1] for i in range(10)])
        y = np.array([i for i in range(10)])
        
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Test train-test split sizes
        assert len(X_test) == int(0.2 * len(X)), "Test size is incorrect."
        assert len(X_train) == len(X) - len(X_test), "Train size is incorrect."
        
        # Test correspondence between X and y
        assert X_train.shape[0] == y_train.shape[0], "Mismatch between X_train and y_train."
        assert X_test.shape[0] == y_test.shape[0], "Mismatch between X_test and y_test."

        # Test reproducibility with random_state
        X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X, y, test_size=test_size, random_state=42)
        np.testing.assert_array_equal(X_train, X_train_2, "Random state does not ensure reproducibility.")
        np.testing.assert_array_equal(X_test, X_test_2, "Random state does not ensure reproducibility.")
        np.testing.assert_array_equal(y_train, y_train_2, "Random state does not ensure reproducibility.")
        np.testing.assert_array_equal(y_test, y_test_2, "Random state does not ensure reproducibility.")
