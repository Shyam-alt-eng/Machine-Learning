
    # Linear Regression From Scratch (Univariate)

This project implements "univariate linear regression from scratch " using Python, NumPy, and Matplotlib without relying on machine learning libraries such as `scikit-learn`.

The purpose of this implementation is to understand the mathematical foundations, gradient descent optimization, and evaluation metrics behind linear regression.

---

## Model Overview

This implementation supports only univariate datasets, where:
-> There is one input feature (X)
-> One target variable (y)

The model learns a straight-line relationship:

y = m*x + b

Where:
- m → slope (weight)
- b → intercept (bias)

---

## Dataset Used
Salary vs Years of Experience

- Feature (X): Years of Experience  
- Target (y): Salary (in thousands)

This is a real-world inspired dataset commonly used to demonstrate linear regression behavior.

## Algorithm Details

### Optimization
- Gradient Descent
- Iteratively updates parameters `m` and `b` to minimize loss

### Loss Function
- Mean Squared Error (MSE), used for optimization during training

### Evaluation Metrics
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score (Coefficient of Determination)

## Mathematical Formulation

### Prediction
ŷ = m*x + b

### Mean Squared Error
MSE = (1/n) Σ (y − ŷ)²

### Root Mean Squared Error
RMSE = √MSE

### R² Score
R² = 1 − (RSS / TSS)

Where:
- RSS(Residual Sum of Square) = Σ (y − ŷ)²
- TSS(Total Sum of Square) = Σ (y − ȳ)²

---

## Training Details

- Learning Rate: 0.01  
- Epochs: 1000  
- Parameters initialized as:
  - m = 0.0
  - b = 0.0

Training progress is logged every 100 epochs.

## Output

After training, the program prints:
- Final slope (m)
- Final intercept (b)
- MSE
- RMSE
- R² score

It also visualizes:
- Initial untrained line
- Final regression line fitted to the data

## How to Run

1. Install dependencies:
pip install numpy matplotlib

python main.py
