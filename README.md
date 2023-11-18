
# Gradient Descent for Multi-Linear Regression

## Introduction

This project presents a Python implementation of the gradient descent algorithm for multi-linear regression. Designed to handle problems with 'r' predictors, it allows customization of the learning rate (η) and the number of iteration steps. The implementation is tested on two datasets: `advertising.csv` and `auto.csv`, using a suitable train-test split to evaluate the model's performance.

## Features

- Implements gradient descent for multi-linear regression problems.
- Customizable learning rate and iteration steps.
- Evaluation using cost function and R-squared test.
- Tested on real-world datasets (`advertising.csv` and `auto.csv`).
- Visualization tools for analyzing regression results.


### Prerequisites

- Python environment 
- Libraries: NumPy, Matplotlib, Seaborn (optional), Scikit-learn.

### Installation

Clone the repository to your local machine:

```bash
git clone [repository-url]
```

### Usage

1. Navigate to the project directory.
2. Open the provided Jupyter notebooks (`main_advertising.ipynb` and `main_auto.ipynb`) to see the implementation on the respective datasets.
3. Modify the parameters (learning rate, iterations, test size) in the `Model` class instantiation as needed.



## Notebooks for Analysis

Two Jupyter notebooks are provided:

1. `advertising_analysis.ipynb` for the `advertising.csv` dataset.
2. `auto_analysis.ipynb` for the `auto.csv` dataset.

These notebooks guide you through the process of loading the data, creating an instance of the `Model` class, running the regression analysis, and visualizing the results.
