# Machine Learning Foundations: Linear & Logistic Regression

## Overview

This document reviews the core mathematical concepts and Python implementations of **Linear Regression** and **Logistic Regression**, which are fundamental algorithms in machine learning.

---

## Core Math Concepts

### 1. Vectors and Dot Product

- Data and model parameters are represented as vectors.
- The dot product is the weighted sum of features:

$$
\mathbf{w} \cdot \mathbf{x} = \sum_{i=1}^n w_i x_i
$$


---

### 2. Linear Function (for Regression)

- Linear prediction is a weighted sum plus bias:

$$
\hat{y} = \mathbf{w} \cdot \mathbf{x} + b
$$

---

### 3. Sigmoid Function (for Classification)

- Maps real values into probabilities between 0 and 1:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

---

### 4. Loss Functions

- **Mean Squared Error (MSE):** For linear regression

$$
MSE = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i)^2
$$

- **Cross-Entropy Loss:** For logistic regression (binary classification)

$$
J = - \frac{1}{m} \sum_{i=1}^m \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1 - \hat{y}_i) \right]
$$

    where m is the number of training samples.

- **Loss function (Binary Cross-Entropy):**

The goal is to maximize the likelihood of correct classification, equivalently minimize the loss:

$$
J(w,b)=\frac{−1}{m}\sum_{i=1}^m [y^{(i)}log⁡p^{(i)}+(1−y^{(i)})log⁡(1−p^{(i)})]
$$

---

### 5. Logistic Regression
Objective:

Predict the probability of a binary outcome $$y∈{0,1}$$ given input features x.
#### Model:

First compute a linear combination:
$$ z=w⋅x+b $$

Then apply the sigmoid function to map z to a probability $$p∈(0,1)$$

$$
p=σ(z)=\frac{1}{1+e^{-z}}
$$


### 6. Optimization: Gradient Descent

To minimize J, compute gradients:

$$
\frac{∂J}{∂wj}=\frac{2}{m}\sum_{i=1}^m \left( \hat{y}^i + {y}^i  \right)x_j^i
$$
$$
\frac{∂J}{∂b}=\frac{2}{m}\sum_{i=1}^m \left( \hat{y}^i + {y}^i  \right)
$$

Update parameters iteratively:
$$
w_j:=w_j-α \frac{∂J}{∂w_j}
$$
$$
b:=b-α \frac{∂J}{∂b}
$$

where α is the learning rate.



