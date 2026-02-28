import numpy as np

class LogisticRegressionScratch:
    
    def __init__(self, learning_rate=0.01, epochs=10000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, y, y_hat):
        m = y.shape[0]
        epsilon = 1e-15  # for avoiding log(0)
        return -(1/m) * np.sum(
            y * np.log(y_hat + epsilon) +
            (1 - y) * np.log(1 - y_hat + epsilon)
        )
    
    def fit(self, X, y):
        m, n = X.shape
        
        self.w = np.zeros((n, 1))
        self.b = 0
        
        for i in range(self.epochs):
            
            z = np.dot(X, self.w) + self.b
            y_hat = self.sigmoid(z)
            
            dw = (1/m) * np.dot(X.T, (y_hat - y))
            db = (1/m) * np.sum(y_hat - y)
            
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            if i % 1000 == 0:
                loss = self.compute_loss(y, y_hat)
                print(f"Epoch {i}, Loss: {loss:.6f}")

    def predict_proba(self, X):
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)
        
    def predict(self, X):
        y_hat = self.predict_proba(X)
        return (y_hat >= 0.5).astype(int)
