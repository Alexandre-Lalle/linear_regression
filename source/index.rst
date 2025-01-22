.. Regression Module documentation master file, created by
   sphinx-quickstart on Mon Jan 20 22:05:51 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Regression Module Documentation!
================================================

Overview
--------
The `regression` Python module provides robust tools for linear regression modeling, data preprocessing, and hyperparameter grid search optimization.

Documentation Contents
-----------------------
The documentation is divided into several sections: 

- **Linear Regression**: Implementation and usage of the linear regression model.
- **Data Preprocessing**: Tools to normalize, clean, and split datasets into training and testing subsets.
- **Hyperparameter Optimization**: Grid search functions for optimizing model parameters.

Installation
------------
To install the package, use the following command:

.. code-block:: bash

    python setup.py install

Getting Started
---------------
To get started with the Linear Regression Package, follow these steps:

1. Install the package using pip.
2. Prepare your data.
3. Initialize and configure the model.
4. Fit the model to your training data.
5. Evaluate the model using test data.

Usage
-----
Here is a simple example of how to use the Linear Regression Package:

.. code-block:: python

    from regression import LinearRegression

    # Initialize model
    model = LinearRegression(
        learning_rate=0.01,
        max_iter=1000,
        normalize=True,
        l1_ratio=0.5,
        l2_ratio=0.5
    )

    # Fit model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Get R² score
    score = model.score(y_test, y_pred)

API Reference
-------------
The API section provides detailed documentation of the classes and methods available in the package.

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   modules/regression
   modules/preprocessing



