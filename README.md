# Linear Regression Module

The **Regression Module** is a Python package offering robust tools for implementing and analyzing linear regression models, combined with data preprocessing and hyperparameter optimization techniques.

## Key Features

- **Multiple Linear Regression**: A custom implementation of linear regression with gradient descent optimization.
- **Regularization**: Supports L1 (Lasso), L2 (Ridge), and ElasticNet regularization for enhanced model performance.
- **Data Preprocessing**: Built-in functions for feature normalization, outlier removal, and efficient train-test splitting.
- **Visualization Tools**: Plot learning curves and feature importance for better interpretability of your models.
- **Hyperparameter Optimization**: Includes grid search capabilities to fine-tune model parameters effectively.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Alexandre-Lalle/linear_regression.git
   cd linear_regression
   ```
2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Install the package:
    ```bash
    python setup.py install
    ```

## Usage
```python
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
```

## Documentation
A comprehensive Sphinx documentation is available for this module, providing detailed information on usage, classes, and functions.

To generate the HTML documentation, run:
```bash
make html
```

## Running Tests
To run the tests using Pytest, execute:
```bash
pytest tests/
```

## License
MIT License

## Author
[Alexandre LALLE](https://github.com/Alexandre-Lalle)