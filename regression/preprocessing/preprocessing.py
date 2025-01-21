#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np
from typing import Tuple

def normalize_features(X: np.ndarray) -> np.ndarray:
    """
    Normalize features by removing the mean and scaling to unit variance.

    Parameters
    ----------
    X : np.ndarray
        The input features to be normalized.

    Returns
    -------
    np.ndarray
        The normalized features.
    """
        
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)


def remove_outliers(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers based on the IQR (Interquartile Range) method.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)

    Returns
    -------
    X_clean, y_clean : Tuple of arrays without outliers
    """
    Q1 = np.percentile(y, 25)  # 25th percentile (lower quartile)
    Q3 = np.percentile(y, 75)  # 75th percentile (upper quartile)
    IQR = Q3 - Q1              # Interquartile range
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Create mask for values within the IQR bounds
    mask = (y >= lower_bound) & (y <= upper_bound)

    return X[mask], y[mask]

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = None) -> tuple:
    """
    Split arrays or matrices into random train and test subsets.

    Parameters
    ----------
    X : np.ndarray
        The input features.
    y : np.ndarray
        The target values.
    test_size : float, optional
        The proportion of the dataset to include in the test split (default is 0.2).
    random_state : int, optional
        Controls the shuffling applied to the data before applying the split (default is None).

    Returns
    -------
    tuple
        A tuple containing the train-test split of inputs and targets: (X_train, X_test, y_train, y_test).
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    
    X = X[indices]
    y = y[indices]
    
    if isinstance(test_size, float):
        test_size = int(X.shape[0] * test_size)
    
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    return X_train, X_test, y_train, y_test