# Automated-Disaster-Message-Classification-using-Machine-Learning

# Disaster Text Classification using TF-IDF and Decision Tree
## Project Overview

This project implements a machine learning–based text classification system that identifies whether a given text message is disaster-related or non-disaster-related.
The solution converts unstructured text into structured numerical features using TF-IDF vectorization and applies a Decision Tree classifier to generate predictions.

The project emphasizes clean feature engineering, model interpretability, and deployment readiness, making it suitable for real-world information filtering and monitoring scenarios.

## Problem Statement

During natural disasters and emergency situations, large volumes of textual data are generated through alerts, reports, and public communications. Manually filtering and identifying disaster-relevant messages is inefficient and time-critical.

The objective of this project is to:

Automatically classify text messages based on disaster relevance

Reduce manual effort in identifying critical information

Enable faster prioritization during emergency and crisis scenarios

## Key Challenges Addressed
### 1. Unstructured Text Data

Presence of punctuation, symbols, and inconsistent formatting

Short and ambiguous messages with limited context

High vocabulary variation across messages

### 2. Feature Engineering Complexity

Converting raw text into numerical representations suitable for ML

Managing high-dimensional, sparse TF-IDF feature spaces

Preserving meaningful terms while minimizing noise

### 3. Model Selection & Generalization

Selecting a model that balances interpretability and performance

Avoiding overfitting on sparse textual features

Ensuring consistent preprocessing during training and inference

## Project Objectives

Clean and standardize raw text data

Transform text into numerical features using TF-IDF

Train and evaluate multiple machine learning models

Select the best-performing model using evaluation metrics

Serialize trained artifacts for reuse and future deployment

## Dataset Description

Data Type: Text dataset

Classification Task: Binary classification

Input Feature: Text messages

Target Variable: Disaster relevance label

Source: Publicly available disaster-related text dataset

## System Workflow
Raw Text Data
   ↓
Text Cleaning & Normalization
   ↓
TF-IDF Feature Vectorization
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Serialization

## Text Processing & Feature Engineering

The following preprocessing steps were applied:

Conversion of text to lowercase

Removal of punctuation and special characters

Transformation of text into numerical features using TF-IDF vectorization

No external NLP libraries were used; feature extraction was handled entirely through Scikit-learn’s TF-IDF Vectorizer.

## Model Development
### Selected Model

Algorithm: Decision Tree Classifier

Learning Type: Supervised Machine Learning

Rationale for Decision Tree

Interpretable decision-making logic

Ability to capture non-linear feature relationships

Strong baseline performance on TF-IDF features

Fast inference suitable for deployment scenarios

Model Artifacts

dt_disaster_text_data_project_best.pkl – trained Decision Tree model

tfidf_vectorizer_disaster.pkl – TF-IDF feature extractor

These artifacts ensure reproducible and consistent inference across environments.

## Model Evaluation

Model performance was evaluated using:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

These metrics provide a balanced assessment of classification effectiveness.

## Technology Stack

Programming Language: Python

Feature Engineering: TF-IDF (Scikit-learn)

Machine Learning: Decision Tree Classifier

Model Persistence: Pickle

## Repository Structure
├── TEXT DATA PROJECT.ipynb
├── dt_disaster_text_data_project_best.pkl
├── tfidf_vectorizer_disaster.pkl
├── requirements.txt
├── README.md

## Key Outcomes

Built a reliable text classification pipeline using TF-IDF features

Identified Decision Tree as the best-performing model

Created reusable model and vectorizer artifacts

Demonstrated practical application of machine learning on text data

## Future Enhancements

Compare performance with Logistic Regression, Random Forest, and SVM

Handle class imbalance using resampling techniques

Deploy as a real-time web or API-based application

Add probability-based confidence scores



# Project Links

Hugging Face Live Demo (optional): **https://huggingface.co/spaces/venugopal99Bathula/Disaster_Message_Classification**

# Author

Bathula Venu Gopal
Data Science Intern @ Innomatics Research Labs (Batch 419)
Former Amazon ML Data Associate
Focus Areas: Machine Learning & Data Analytics
