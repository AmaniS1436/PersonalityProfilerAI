# PersonalityProfilerAI: Classification with ML, SHAP, PCA, and Noise Robustness

## Overview

**PersonalityProfilerAI** is a machine learning project focused on predicting synthetic personality types using structured data, exploratory data analysis, multiple classification algorithms, SHAP explanations, PCA-based visualization, and noise resilience testing.

The model not only performs high-accuracy classification across several algorithms (Logistic Regression, Random Forest, XGBoost, etc.), but it also evaluates the robustness of these models under noisy conditions and interprets their decisions using SHAP.

---

## Features

- Exploratory Data Analysis (EDA) for understanding feature distributions and correlations.
- Multi-model comparison: Logistic Regression, Random Forest, Decision Tree, SVM, Naive Bayes, and XGBoost.
- SHAP (SHapley Additive exPlanations) for feature importance and explainability.
- Noise injection to evaluate model robustness.
- PCA (Principal Component Analysis) for dimensionality reduction and 2D visualization.
- Inference example for real-time personality prediction.

---

## Dataset

A synthetic dataset of personality types with numerical features like:

- `social_energy`
- `online_social_usage`
- `emotional_stability`
- `neuroticism`
- ... and other psychometric indicators.

Target variable: `personality_type` (multi-class categorical)

**Data Source:** [Synthetic Personality Dataset on Kaggle](https://www.kaggle.com/datasets/miadul/introvert-extrovert-and-ambivert-classification)

---

## Main Steps

### 1. Data Loading & Cleaning

- Handles missing values by imputing mean values.
- Encodes categorical target using `LabelEncoder`.

### 2. Exploratory Data Analysis

- Class distribution plots
- Histograms for each feature
- Heatmap of correlations

### 3. Model Training

- Compares six models using accuracy, F1, precision, recall, and ROC AUC.
- Outputs classification reports and confusion matrices.

### 4. Robustness Testing

- Adds Gaussian noise to features
- Re-evaluates model performance on noisy test data
- Plots performance drop (if any)

### 5. SHAP Explainability

- Uses `TreeExplainer` to interpret XGBoost decisions
- Visualizes feature importance

### 6. PCA Visualization

- Reduces feature space to 2D
- Plots classes on PCA-reduced axes to visualize separability

### 7. Inference

- Predicts the personality type of a single sample using SVM.

---

## Results

- **XGBoost** and **Random Forest** achieved top accuracy on clean data.
- Most models maintained high performance even after noise injection.
- SHAP and PCA helped validate the logic and separability of features.

---

## Conclusion

This project showcases a comprehensive pipeline for synthetic personality type prediction. From thorough EDA to multi-model benchmarking, explainability through SHAP, noise robustness testing, and PCA-based visualization, **PersonalityProfilerAI** serves as a strong foundation for personality classification tasks. The methods implemented ensure both accuracy and interpretability — two critical pillars in applied machine learning.
