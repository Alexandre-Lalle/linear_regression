#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np
from itertools import product
from .linear_regression import LinearRegression

def grid_search_lr(X_train, y_train, X_val, y_val, param_grid):
    """
    Perform a grid search to find the best hyperparameters for LinearRegression.
    
    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training data.
    y_train : array-like of shape (n_samples,)
        Training target values.
    X_val : array-like of shape (n_samples, n_features)
        Validation data.
    y_val : array-like of shape (n_samples,)
        Validation target values.
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values. 
        Parameters not specified will use default values.
        
    Returns
    -------
    best_params : dict
        Best parameter combination found.
    best_score : float
        Best R² score achieved.
    """
    
    # Default parameters
    default_params = {
        'learning_rate': 0.001,
        'max_iter': 3000,
        'tol': 1e-6,
        'normalize': False,
        'l1_ratio': 0.0,
        'l2_ratio': 0.0,
        'remove_outliers': False
    }
    
    best_score = -np.inf
    best_params = None
    
    # Only use parameters that are in param_grid
    keys, values = zip(*[(k, v) for k, v in param_grid.items()])
    
    for v in product(*values):
        # Start with default parameters
        current_params = default_params.copy()
        
        # Update with parameters being tested
        current_params.update(dict(zip(keys, v)))
        
        # Create and train model
        model = LinearRegression(**current_params)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_val)
        score = model.score(y_val, y_pred)
        
        if score > best_score:
            best_score = score
            best_params = current_params
    
    return best_params, best_score
