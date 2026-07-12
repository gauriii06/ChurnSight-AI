# ChurnSight AI

**An end-to-end customer churn prediction and retention intelligence platform powered by explainable machine learning, customer lifetime value analysis, and actionable retention recommendations.**

## Overview
Customer retention is a critical business challenge, as acquiring new customers is often more expensive than retaining existing ones. ChurnSight AI is an intelligent analytics platform designed to help businesses identify customers at risk of leaving and understand the reasons behind those predictions.
The platform combines machine learning, explainable AI, customer lifetime value (CLV) analysis, and retention recommendations within a single web application. Unlike traditional churn prediction models that provide only a binary outcome, ChurnSight AI explains each prediction using SHAP values and generates actionable insights to support data-driven decision-making.
The application provides an interactive dashboard for analyzing customer behavior, performing batch predictions, and helping organizations proactively reduce customer churn.

---

## Features

* Predict customer churn using supervised machine learning models.
* Compare multiple classification algorithms, including Logistic Regression, Decision Tree, and Random Forest.
* Explain model predictions using SHAP (SHapley Additive exPlanations).
* Generate personalized customer retention recommendations based on churn drivers.
* Perform Customer Lifetime Value (CLV) analysis to identify high-value customers.
* Batch prediction support through CSV uploads.
* Interactive analytics dashboard for business insights.
* Secure authentication with role-based access control.
* Customer search, filtering, and prediction history.
* Responsive Flask-based web interface with integrated database support.

---

## Tech Stack

**Programming Languages**

* Python
* SQL

**Machine Learning**

* Scikit-learn
* Pandas
* NumPy
* SHAP

**Backend**

* Flask

**Database**

* MySQL

**Frontend**

* HTML
* CSS
* JavaScript
* Bootstrap

**Development Tools**

* Git
* GitHub
* VS Code

---

## Architecture

The system follows a modular client-server architecture.

1. Users interact with the Flask web application through a browser.
2. Customer data is validated and stored in a MySQL database.
3. The preprocessing module cleans and transforms incoming data before prediction.
4. The machine learning engine loads the trained classification model and generates churn predictions.
5. SHAP generates feature-level explanations describing the factors influencing each prediction.
6. The recommendation engine analyzes churn drivers and produces personalized retention strategies.
7. Customer Lifetime Value (CLV) analysis identifies high-value customers to support business prioritization.
8. Results are displayed through interactive dashboards and detailed prediction reports.

---

## Machine Learning Pipeline

1. Data Collection

   * Import customer demographic, behavioral, and transactional data.

2. Data Preprocessing

   * Handle missing values.
   * Encode categorical variables.
   * Normalize numerical features.
   * Split the dataset into training and testing sets.

3. Model Training

   * Train multiple classification models including Logistic Regression, Decision Tree, and Random Forest.

4. Model Evaluation

   * Compare model performance using accuracy, precision, recall, F1-score, and confusion matrix.

5. Prediction

   * Predict the probability of customer churn for new or existing customers.

6. Explainability

   * Use SHAP to identify feature contributions and explain individual predictions.

7. Business Intelligence

   * Estimate Customer Lifetime Value (CLV).
   * Generate retention recommendations based on churn risk and customer value.

8. Deployment

   * Serve predictions and analytics through a Flask-based web application with database integration.
