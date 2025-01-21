#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from .preprocessing import remove_outliers, normalize_features

class LinearRegression:
    """
    A multiple linear regression model implementing gradient descent optimization.
    
    Parameters
    ----------
    learning_rate : float, default=0.001
        The learning rate for gradient descent
    max_iter : int, default=1000
        Maximum number of iterations for gradient descent
    tol : float, default=1e-4
        Tolerance for the stopping criterion
    l1_ratio : float, default=0.0
        L1 regularization strength (Lasso)
    l2_ratio : float, default=0.0
        L2 regularization strength (Ridge)
    normalize : bool, default=False
        Whether to normalize features before fitting
    remove_outliers : bool, default=False
        Whether to remove outliers before fitting


    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients for the linear regression problem
    intercept_ : float
        Independent term in the linear model
    n_iter_ : int
        Number of iterations run
    cost_history_ : list
        Cost function value at each iteration
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        max_iter: int = 1000,
        tol: float = 1e-4,
        l1_ratio: float = 0.0,
        l2_ratio: float = 0.0,
        normalize: bool = False,
        remove_outliers: bool = False 
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.normalize = normalize
        self.remove_outliers = remove_outliers
        
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = 0
        self.cost_history_ = []
            
    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the cost function with optional L1 and L2 regularization.
    
        The cost function used is the Mean Squared Error (MSE), with optional
        L1 (Lasso) and L2 (Ridge) regularization terms added. 
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features.
        y : np.ndarray of shape (n_samples,)
            Target values.
        
        Returns
        -------
        cost : float
        The computed value of the cost function, including regularization.
        """
        m = len(y)
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        
        # Add regularization terms
        l1_term = self.l1_ratio * np.sum(np.abs(self.coef_))
        l2_term = self.l2_ratio * np.sum(self.coef_ ** 2)
        
        return mse + l1_term + l2_term
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the linear model using gradient descent.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.array(X)
        y = np.array(y)
        
        if self.remove_outliers:
            X, y = remove_outliers(X, y)
        
        if self.normalize:
            X = normalize_features(X)
        
        m, n = X.shape
        
        # Initialize parameters
        np.random.seed(42)
        self.coef_ = np.random.randn(n) * 0.01
        self.intercept_ = np.random.randn() * 0.01
        
        prev_cost = float('inf')
        self.cost_history_ = []
        
        for iteration in range(self.max_iter):
            # Compute predictions
            y_pred = self.predict(X)
            
            # Compute gradients
            dw = (1/m) * X.T.dot(y_pred - y)
            db = (1/m) * np.sum(y_pred - y)
            
            # Add regularization gradients
            dw += self.l1_ratio * np.sign(self.coef_)
            dw += 2 * self.l2_ratio * self.coef_
            
            # Update parameters
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db
            
            # Compute cost and check convergence
            current_cost = self._compute_cost(X, y)
            self.cost_history_.append(current_cost)
            
            if abs(prev_cost - current_cost) < self.tol:
                break
                
            prev_cost = current_cost
            self.n_iter_ = iteration + 1
            
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values
            
        Warnings
        --------
        If `normalize` is set to False during `fit`, this method does not
        automatically apply feature normalization to the input `X`. Ensure that
        the features of `X` match the scale used during training to avoid
        inconsistent predictions. If normalization was applied during training,
        `X` will be normalized automatically here.
        """
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model not fitted. Call 'fit' first.")

        X = np.array(X)
        if self.normalize:
            X = normalize_features(X)
        return X.dot(self.coef_) + self.intercept_
    
    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.
        
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True values
        y_pred : array-like of shape (n_samples,)
            Predicted values
            
        Returns
        -------
        score : float
            R^2 score
        """
        if len(y_true) < 2:
            raise ValueError("R² score is not well-defined with less than two samples.")
     
        u = ((y_true - y_pred) ** 2).sum()
        v = ((y_true - y_true.mean()) ** 2).sum()
        if v == 0:
            return 0.0 if u == 0 else -float('inf')
        
        return 1 - u/v
    
    def plot_convergence(self):
        """
        Plot the convergence of the cost function during gradient descent.
    
        This method visualizes the value of the cost function over iterations, 
        allowing you to assess whether the model is converging properly.
        """
        if not hasattr(self, 'cost_history_'):
            raise ValueError("Cost history not available. Ensure the model has been trained.")
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.cost_history_)), self.cost_history_)
        plt.title('Convergence of Cost Function')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()
        
    def plot_feature_importance(self, feature_names: Optional[list] = None) -> None:
        """
        Plot the absolute values of feature coefficients as a measure of importance.
        
        Parameters
        ----------
        feature_names : list, optional
            Names of the features (default: None)
        """
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(len(self.coef_))]
            
        plt.figure(figsize=(10, 6))
        importance = np.abs(self.coef_)
        plt.bar(feature_names, importance)
        plt.xticks(rotation=45)
        plt.xlabel('Features')
        plt.ylabel('|Coefficient|')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()