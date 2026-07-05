# Logistic Regression from Scratch

This project implements **Logistic Regression from scratch using NumPy**.

The model is trained on the **Breast Cancer Wisconsin dataset** from Scikit-learn and predicts whether a sample is benign or malignant.

---

## What This Project Covers

* Logistic Regression
* Sigmoid function
* Gradient Descent
* Weight and bias updates
* Binary classification
* Accuracy evaluation

---

## Technologies Used

* Python
* NumPy
* Scikit-learn
* tqdm
* Matplotlib

---

## How It Works

The model first calculates a linear equation:

```python
z = X · weights + bias
```

Then it applies the sigmoid function:

```python
prediction = 1 / (1 + exp(-z))
```

If the prediction is greater than `0.5`, the model returns class `1`; otherwise, it returns class `0`.

---

## Installation

```bash
pip install numpy matplotlib tqdm scikit-learn
```

---

## Run the Project

```bash
python logistic_regression.py
```

---

## Example Output

```text
Loss: 0.69
Loss: 0.45
Loss: 0.32
Accuracy: 0.92
```

---

## Learning Goal

The purpose of this project is to understand how Logistic Regression works internally instead of only using a ready-made library.
