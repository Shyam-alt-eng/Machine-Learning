# Linear Regression From Scratch (Univariate & Multivariate)

This project implements **linear regression from scratch** using Python and NumPy, without relying on machine learning libraries such as `scikit-learn`.

The goal is to deeply understand:
- Linear regression mathematics
- Gradient descent optimization
- Matrix operations
- Evaluation metrics
- Feature scaling

---

## Part 1: Univariate Linear Regression

### Model Overview

This implementation supports **univariate datasets**, where:
- One input feature (X)
- One target variable (y)

The learned relationship is:

y = m·x + b

Where:
- m → slope (weight)
- b → intercept (bias)

---

### Dataset Used

**Salary vs Years of Experience**

- Feature (X): Years of Experience  
- Target (y): Salary (in thousands)

This dataset is commonly used to demonstrate linear regression behavior.

---

### Algorithm Details

#### Optimization
- Gradient Descent
- Iterative parameter updates to minimize loss

#### Loss Function
- Mean Squared Error (MSE)

#### Evaluation Metrics
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score (Coefficient of Determination)

---

### Mathematical Formulation

#### Prediction
ŷ = m·x + b

#### Mean Squared Error
MSE = (1/n) Σ (y − ŷ)²

#### Root Mean Squared Error
RMSE = √MSE

#### R² Score
R² = 1 − (RSS / TSS)

Where:
- RSS = Σ (y − ŷ)²  
- TSS = Σ (y − ȳ)²  

---

### Training Details

- Learning Rate: 0.01  
- Epochs: 1000  
- Initial values:
  - m = 0.0
  - b = 0.0  

Training loss is logged every 100 epochs.

---

### Output

After training, the program prints:
- Final slope (m)
- Final intercept (b)
- MSE
- RMSE
- R² score

It also visualizes:
- Initial untrained line
- Final regression line

---

## Part 2: Multivariate Linear Regression

### Model Overview

This implementation extends linear regression to handle **multiple input features**.

The learned equation becomes:

y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

Where:
- w → weight vector
- b → bias

---

### Matrix Representation

- Feature Matrix: X ∈ ℝⁿˣᵈ  
- Weight Vector: w ∈ ℝᵈ  

#### Prediction
ŷ = X · w + b

---

### Feature Scaling

All features are standardized using **Z-score normalization**:

X_scaled = (X − mean(X)) / std(X)

Feature scaling:
- Improves convergence speed
- Prevents dominance of large-valued features
- Stabilizes gradient descent

---

### Gradient Descent Formulation

#### Gradients
dw = (−2/n) · Xᵀ · (y − ŷ)  
db = (−2/n) · Σ (y − ŷ)

#### Updates
w = w − α·dw  
b = b − α·db  

Where:
- α → learning rate
- n → number of samples

---

### Evaluation Metrics

The multivariate model reports:
- MSE
- RMSE
- R² Score

---

### Important Constraint

The number of features during **prediction must match training**.

Example:
X_train.shape = (n, d)  
X_test.shape  = (m, d)

Mismatch in feature count will cause a matrix dimension error.

---

### Output (Multivariate)

After training, the program outputs:
- Learned weight vector
- Bias
- MSE
- RMSE
- R² score

Visualization is omitted since multivariate data cannot be plotted directly beyond 2D.

---

## How to Run

Install dependencies:
pip install numpy matplotlib

Run the univariate model:
python main.py

Run the multivariate model:
python multivariate_main.py

---

## Learning Outcomes

This project helps understand:
- Linear regression internals
- Gradient descent from scratch
- Matrix multiplication and transpose
- Feature scaling
- Model evaluation metrics
