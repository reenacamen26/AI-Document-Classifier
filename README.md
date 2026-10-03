# AI Document Classification System

## Overview

The AI Document Classification System is a machine-learning-based application that automatically classifies documents into different categories.

## Features

- Upload TXT and PDF documents
- Classify documents using a trained Machine Learning model
- Display prediction confidence
- Analyze dataset documents
- Search documents by name
- Filter documents by category
- Extract text from uploaded documents
- Generate document summaries
- View detailed document analysis
- Interactive Streamlit dashboard

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Joblib
- PyPDF2
- Machine Learning
- TF-IDF Vectorization

## Document Categories

The system supports:

- Bank Statement
- Certificate
- Invoice
- Research Paper
- Resume

## Features

- Upload TXT and PDF documents
- Classify documents using a trained ML model
- Display prediction confidence
- Analyze multiple dataset documents
- Search documents by name
- Filter documents by category
- Generate document summaries
- Extract text from documents
- Extract invoice information
- Display document statistics and category distribution

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- Joblib
- PyPDF
- NLTK

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt


## Project Structure

```text
AI-Document-Classifier/
│
├── app.py
├── train_model.py
├── generate_dataset.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── dataset/
│   ├── Bank Statement/
│   ├── Certificate/
│   ├── Invoice/
│   ├── Research Paper/
│   └── Resume/
│
└── model/
    └── classifier.pkl


## How to Run

### 1. Install the required packages

```bash
pip install -r requirements.txt