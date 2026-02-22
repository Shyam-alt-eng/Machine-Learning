# This linear regression model is only for univariate dataset

import numpy as np
import matplotlib.pyplot as plt

#  Years of Experiance (X)
X = np.array([
    1.1, 1.3, 1.5, 2.0, 2.2,
    2.9, 3.0, 3.2, 3.7, 3.9,
    4.0, 4.5, 4.9, 5.1, 5.3,
    5.9, 6.0, 6.8, 7.1, 7.9,
    8.2, 8.7, 9.0, 9.5, 10.3
])

# Salary in thousands (y)
y = np.array([
    39, 41, 43, 48, 50,
    57, 60, 63, 67, 69,
    72, 75, 78, 80, 83,
    88, 90, 95, 98, 105,
    108, 110, 112, 118, 125
])

# The model does not know the best line initially, so we put  m = 0.0, b = 0.0
m = 0.0
b = 0.0

#  we give current m,b and returning the predicted value, and does model returns(y^)
def predict(X,m,b):
    return m * X + b

# calculating root mean squared value
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# calculating MSE

def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# gradient to minimizwe the mse
def gradients(X,y_true,y_pred):
    n = len(X)
    dm = (-2/n) * np.sum(X * (y_true - y_pred))
    db = (-2/n) * np.sum(y_true - y_pred)
    return dm, db

learning_rate = 0.01
epochs = 1000

 # initial plotting before regression
plt.scatter(X,y,label = "ACtual Data")
plt.plot(X,predict(X,m,b), label="Unregressed line")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# R2 value of the model 
def r2(y_true, y_pred):
    RSS = np.sum((y_true - y_pred) ** 2)
    TSS = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (RSS/TSS)

for i in range(epochs):
    y_pred = predict(X,m,b)

    dm,db = gradients(X,y,y_pred)

    m = m - learning_rate * dm
    b = b - learning_rate * db

    if i % 100 == 0:
        print(f"Epoch {i} | Loss: {MSE(y, y_pred):.4f}")

print("\nTraining complete!")
print(f"Final slope (m): {m}")
print(f"Final intercept (b): {b}")
print(f"MSE : {MSE(y, y_pred)}")
print(f"RMSE : {RMSE(y, y_pred)}")

print(f"R2 = {r2(y,y_pred)}")

 # plotting after regression
plt.scatter(X, y, label="Actual Data")
plt.plot(X, predict(X, m, b), label="Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary (in thousands)")
plt.legend()
plt.show()