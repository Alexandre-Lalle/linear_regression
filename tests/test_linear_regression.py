import numpy as np
import pytest
from regression import LinearRegression

class TestLinearRegression:
    """
    Test class for the linear regression model.
    """
    
    @pytest.fixture
    def sample_data(self):
        # Create simple synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = 2 * X[:, 0] + 0.5 * X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1
        return X, y
    
    def test_init(self):
        model = LinearRegression()
        assert model.learning_rate == 0.001
        assert model.max_iter == 1000
        assert model.tol == 1e-4
        assert model.l1_ratio == 0.0
        assert model.l2_ratio == 0.0
        
    def test_fit_predict(self, sample_data):
        X, y = sample_data
        model = LinearRegression(normalize=True)
        
        # Test unfitted model
        with pytest.raises(ValueError, match="Model not fitted. Call 'fit' first."):
            model.predict(X)
        
        model.fit(X, y)
        
        # Check coefficients shape
        assert model.coef_.shape == (X.shape[1],)
        assert isinstance(model.intercept_, float)
        
        # Test predictions
        y_pred = model.predict(X)
        assert y_pred.shape == y.shape
        
    def test_score(self, sample_data):
        X, y = sample_data
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        score = model.score(y, y_pred)
        assert 0 <= score <= 1
                
    def test_regularization(self, sample_data):
        X, y = sample_data
        
        # Test L1 (Lasso)
        model_l1 = LinearRegression(l1_ratio=1.0)
        model_l1.fit(X, y)
        
        # Test L2 (Ridge)
        model_l2 = LinearRegression(l2_ratio=1.0)
        model_l2.fit(X, y)
        
        # Coefficients should be different
        assert not np.allclose(model_l1.coef_, model_l2.coef_)
                
    def test_input_validation(self):
        model = LinearRegression()
        
        # Test invalid inputs for fit method
        with pytest.raises(ValueError):
            model.fit(np.array([[1]]), np.array([1, 2]))

