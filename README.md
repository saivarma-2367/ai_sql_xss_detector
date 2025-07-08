# AI_SQL_XSS_DETECTOR

This project implements an AI-powered detector for SQL Injection and Cross-Site Scripting (XSS) vulnerabilities.

## Project Structure

The project is organized into the following main directories and files:

AI_SQL_XSS_DETECTOR/
├── api/
│   ├── app.py
├── data/
│   ├── filtered_SQL_XSS_only.csv
│   ├── raw_dataset.csv
│   └── test_input.xlsx
├── ml_model/
│   ├── label_encoder.pkl
│   ├── rf_model_compressed.pkl
│   ├── train.py
│   └── vectorizer.pkl
├── test/
│   └── evaluate_model_accuracy.py
├── utils/
│   ├── pycache/
│   ├── preprocess.py
│   └── environment.md
├── .
├── INTEL_Project_Documentation.pdf
├── model_architecture.png
├── model_report.xlsx
├── problem_scope_document.md
├── README.md
├── requirements.txt
└── Working Methodology of the SQLI_XSS Detector Model.mp4


## Getting Started

To get this project up and running, follow these steps:

### 1. Prerequisites

Ensure you have Python (3.11 recommended) installed on your system.

### 2. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/saivarma-2367/ai_sql_xss_detector
cd AI_SQL_XSS_DETECTOR
pip install -r requirements.txt
3. Environment Setup
For detailed environment setup instructions, refer to:

utils/environment.md

Key Components and How to Access Them
1. Data
Raw Dataset: The original dataset used for training is located at data/raw_dataset.csv.

Filtered Dataset: A processed version of the dataset, likely containing only SQL and XSS relevant entries, can be found at data/filtered_SQL_XSS_only.csv.

Test Input: Example input data for testing purposes is available in data/test_input.xlsx.

2. Machine Learning Model
The core machine learning components are stored in the ml_model/ directory:

Trained Model: The compressed Random Forest model is saved as ml_model/rf_model_compressed.pkl.

Vectorizer: The trained vectorizer (e.g., TF-IDF) used for text feature extraction is ml_model/vectorizer.pkl.

Label Encoder: The encoder used for target labels is ml_model/label_encoder.pkl.

Training Script: The script used to train the model is ml_model/train.py. You can execute this script to retrain the model.

3. API (Streamlit Application)
The project features a user-friendly web interface built with Streamlit:

Streamlit Application: The main application for interacting with the detection model is api/app.py.

To run the Streamlit application: Navigate to the project root directory and execute:

Bash

streamlit run api/app.py
This will typically open the application in your web browser at http://localhost:8501 (or similar).

4. Model Evaluation
Accuracy Evaluation Script: The script for evaluating the model's accuracy is test/evaluate_model_accuracy.py. You can run this script to assess the performance of the trained model.

5. Utilities
Preprocessing Script: Common data preprocessing functions are located in utils/preprocess.py.

Environment Documentation: Detailed environment setup instructions can be found in utils/environment.md.

6. Documentation and Reports
Several key documents provide detailed information about the project:

Project Documentation: Comprehensive documentation of the project is available in INTEL_Project_Documentation.pdf.

Model Report: A detailed report on the model's performance, metrics, and possibly insights can be found in model_report.xlsx.

Problem Scope Document: This document outlines the problem statement and scope of the project: problem_scope_document.md.

Working Methodology: A video of the methodology used and how to run the SQLI/XSS Detector Model is in Working Methodology of the SQLI_XSS Detector Model.mp4.

Model Architecture: A visual representation or description of the model's architecture is in model_architecture.png.

Project Video Demonstration
A brief video demonstrating the functionality and key features of the AI SQL/XSS Detector Model is available here:

https://github.com/saivarma-2367/ai_sql_xss_detector/blob/main/Working%20Methodology%20of%20the%20SQLI_XSS%20Detector%20Model.mp4

https://drive.google.com/file/d/1ihBqbLEA3SyFQCi0MJXlesx_4xPmZ4Tl/view?usp=sharing
