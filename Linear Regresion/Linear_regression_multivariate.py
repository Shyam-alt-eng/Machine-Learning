import numpy as np

class LinearRegressionMultivariate:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.b = 0
        

        # scaling parameters 
        self.mean = None
        self.std = None

    # Scaling function (uses stored mean & std)
    def Sacle(self, X):
        return (X - self.mean) / self.std

    def predict(self, X):
        X = np.array(X)

        # ensure X is 2D for matrix operations
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self.Sacle(X)

        return X_scaled @ self.weights + self.b

    def gradients(self, X, y_true, y_pred, n_samples):
        dw = (-2 / n_samples) * (X.T @ (y_true - y_pred))
        db = (-2 / n_samples) * np.sum(y_true - y_pred)
        return dw, db

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def rmse(self, y_true, y_pred):
        return np.sqrt(self.mse(y_true, y_pred))

    def r2(self, y_true, y_pred):
        RSS = np.sum((y_true - y_pred) ** 2)
        TSS = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (RSS / TSS)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        # compute scaling params ONCE 
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1  # avoid divide-by-zero

        self.loss_history = []  # to store loss values for each epoch

        X_scaled = self.Sacle(X)

        for i in range(self.epochs):
            y_pred = X_scaled @ self.weights + self.b

            dw, db = self.gradients(X_scaled, y, y_pred, n_samples)

            self.weights -= self.learning_rate * dw
            self.b -= self.learning_rate * db

            loss = self.mse(y, y_pred)
            self.loss_history.append(loss)

            if i % 100 == 0:
                print(f"Epoch {i} | Loss: {self.mse(y, y_pred):.4f}")
 
