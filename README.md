# 01ClassifierFromScrath
# Banknote Authentication - Logistic Regression


## Overview

This project implements a functional binary classifier from scratch using **Logistic Regression** to classify banknotes as either genuine or forged. It was developed to understand the core concepts behind binary classification models, especially logistic regression, by diving deep into the mathematical principles behind the model and its implementation.

## Purpose

The purpose of this project is to:
- Build a logistic regression model for binary classification.
- Gain a better understanding of the inner workings of machine learning models.
- Improve the grasp of core concepts by implementing them from scratch.

## Dataset

The data used in this project comes from images taken for the evaluation of an authentication procedure for banknotes. The dataset consists of features extracted from **images** of genuine and forged banknotes.
DATASET from [UCI Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication)

### Dataset Characteristics:
- **Subject Area:** Computer Science
- **Associated Tasks:** Classification
- **Feature Type:** Real
- **Number of Instances:** 1372
- **Number of Features:** 4

### Data Description:

The data consists of 4 features extracted using **Wavelet Transform** from high-resolution grayscale images of banknotes. An industrial camera, usually used for print inspection, was used for digitization. The final images have a resolution of 400x400 pixels with around **660 dpi**. 

### Sample Data:

The dataset is stored in a CSV format where each row represents an instance with its features followed by the target label (0 for forged and 1 for genuine).

| Feature 1 | Feature 2 | Feature 3 | Feature 4 | Label |
|-----------|-----------|-----------|-----------|-------|
| 2.2504    | 3.5757    | 0.35273   | 0.2836    | 0     |
| -1.3971   | 3.3191    | -1.3927   | -1.9948   | 1     |
| 0.39012   | -0.14279  | -0.031994 | 0.35084   | 1     |
| -1.6677   | -7.1535   | 7.8929    | 0.96765   | 1     |

## Requirements

The project requires the following libraries:

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn (for evaluation metrics)

## File Structure

The project is organized as follows:


## How to Run

1. Clone the repository:
2. Navigate to the project folder:
3. Create a virtual environment:
4. Install the required dependencies:
5. Run the model:


## Output

Upon running the `main.py` file, the program splits the data into training, validation, and test sets, then trains a logistic regression model. It also outputs performance metrics, including **Accuracy**, **Precision**, **Recall**, and **F1-Score**.

**Sample Output:**

The model's weights and biases are saved to a file called `banknote_auth_Model.pkl`, which can be loaded for further inference or evaluation.

## Evaluation Metrics

The model is evaluated using the following metrics:
- **Accuracy:** Percentage of correctly classified instances.
- **Precision:** The proportion of positive predictions that were actually correct.
- **Recall:** The proportion of actual positives that were correctly predicted.
- **F1-Score:** A harmonic mean of precision and recall, providing a balanced measure.

## Conclusion

By implementing logistic regression from scratch, this project offers a practical understanding of how binary classifiers function. The training process and evaluation metrics further demonstrate the performance of the model on the banknote authentication task.


