# Chronic Kidney Disease Classification

A comprehensive machine learning project for classifying whether a patient has Chronic Kidney Disease (CKD) or not, using neural networks and logistic regression models.

## Overview

This project implements multiple machine learning approaches to predict the presence of Chronic Kidney Disease based on various clinical and laboratory measurements. The project includes extensive data preprocessing, multiple neural network architectures with hyperparameter tuning, and comparative analysis with logistic regression.

## Dataset

The project uses the `kidney_disease.xlsx` dataset containing:
- **24 clinical and laboratory features** (blood glucose levels, hemoglobin, potassium, specific gravity, albumin, etc.)
- **Target variable:** classification (CKD: 1, Not CKD: 0)
- **Data preprocessing includes:** handling missing values, encoding categorical variables, and feature normalization

## Project Structure

### 1. Data Preprocessing
- Loading and exploratory data analysis (EDA)
- Handling missing values:
  - **Numerical features:** Replaced with median values
  - **Categorical features:** Replaced with mode values
- Encoding categorical variables using LabelEncoder
- Feature normalization using MinMaxScaler (value range: 0-1)
- Correlation analysis and feature visualization

### 2. Neural Network Models

#### Model 1: Neural Network Without Hidden Layer
- Single output layer with sigmoid activation
- Baseline model for comparison

#### Model 2: Neural Network with One Hidden Layer (Variable Hidden Units)
- Hidden layer with different unit sizes: 1, 6, 12, 18, 24, 30, 36, 42, 48
- Fixed sigmoid output layer
- Tests the effect of hidden layer size

#### Model 3: Neural Network with Different Activation Functions
- One hidden layer with 60 units
- Compared activation functions: ReLU, Tanh, Sigmoid, and None
- Sigmoid activation for output layer

#### Model 4: Neural Network with Different Batch Sizes
- One hidden layer with 60 units and ReLU activation
- Tested batch sizes: 4, 32, 64
- Analyzes the impact of batch size on training

#### Model 5: Best Performing Neural Network (Production Model)
- **Architecture:**
  - 100 hidden units with ReLU activation
  - 20% Dropout for regularization
  - Sigmoid output layer
- **Training parameters:**
  - Optimizer: Adam
  - Loss function: Binary Crossentropy
  - Batch size: 32
  - Epochs: 200
- **Evaluation:** K-fold cross-validation (4 folds) and train-test split approach

### 3. Logistic Regression Baseline
- Trained on original features and polynomial-augmented features
- Tested polynomial features of degrees 1-5
- Provides baseline comparison for neural network performance

## Key Features

### Model Evaluation Methods
1. **K-fold Cross Validation:** 4-fold cross-validation for robust performance estimation
2. **Train-Test Split:** Train/validation/test data split for generalization assessment
3. **Metrics:** Accuracy, Loss, Confusion Matrix, Classification Report

### Visualizations
- Training vs. validation loss curves
- Training vs. validation accuracy curves
- Feature distribution histograms
- Categorical feature count plots
- Feature correlation heatmap
- Feature relationship scatter plots

## Results

The best-performing neural network model includes:
- **100 hidden units** with ReLU activation
- **Dropout (0.2)** for regularization
- **200 epochs** of training
- Consistently achieves high accuracy across 4-fold cross-validation

The trained model is saved as `ckd-model.h5` for production deployment.

## Requirements

- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow/keras
- numpy

## Usage

1. **Run the entire notebook:** Execute all cells in `Chronic_Kidney_Disease_Classification.ipynb` to:
   - Load and preprocess data
   - Train multiple neural network models
   - Evaluate different architectures
   - Train and save the best model

2. **Using the pre-trained model:**
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('ckd-model.h5')
   predictions = model.predict(preprocessed_features)
   ```

## Key Findings

- Adding a hidden layer substantially improves model performance
- ReLU and Tanh activation functions are more effective than sigmoid for hidden layers
- Dropout regularization helps prevent overfitting
- Batch size of 32 provides a good balance between training efficiency and accuracy
- The model generalizes well to unseen test data

## Files

| File | Description |
|------|-------------|
| `Chronic_Kidney_Disease_Classification.ipynb` | Main Jupyter |
| `ckd-model.h5` | Pre-trained best-performing neural network model |

## Methodology

The project follows a systematic approach:
1. **Data Understanding**
2. **Data Preprocessing:** Clean, encode, and normalize features
3. **Model Development:** Build and test multiple architectures
4. **Hyperparameter Tuning:** Optimize hidden units, activation functions, batch sizes
5. **Evaluation:** Rigorous cross-validation and test set evaluation
6. **Production Deployment:** Train final model on all data and save for inference

## Notes

- GPU/CPU status is checked at the beginning for optimal training
- All models use binary cross-entropy loss (binary classification problem)
- Adam optimizer is used throughout for consistent convergence
- K-fold cross-validation provides robust performance estimates
- The final model is trained on all available data (production step) after hyperparameter selection

## Author
**Seyyed Hossein Hosseini**