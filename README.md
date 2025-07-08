AI_SQL_XSS_DETECTOR
This project implements an AI-powered detector for SQL Injection (SQLi) and Cross-Site Scripting (XSS) vulnerabilities. It leverages machine learning techniques to identify malicious input patterns and includes a user-friendly interface via a Streamlit web app.

Project Demonstration:
Watch the demo video here:
GitHub: (https://github.com/saivarma-2367/ai_sql_xss_detector/blob/main/Working%20Methodology%20of%20the%20SQLI_XSS%20Detector%20Model.mp4)
Google Drive: (https://drive.google.com/file/d/1ihBqbLEA3SyFQCi0MJXlesx_4xPmZ4Tl/view?usp=sharing)


Project Structure:
AI_SQL_XSS_DETECTOR/
├── api/
│   └── app.py                      # Streamlit web interface
├── data/
│   ├── raw_dataset.csv             # Full original dataset
│   ├── filtered_SQL_XSS_only.csv  # Cleaned dataset with only SQLi and XSS entries
│   └── test_input.xlsx             # Sample inputs for testing
├── ml_model/
│   ├── train.py                    # Training script
│   ├── rf_model_compressed.pkl    # Trained Random Forest model
│   ├── vectorizer.pkl             # Text vectorizer (e.g., TF-IDF)
│   └── label_encoder.pkl          # Label encoder for output classes
├── test/
│   └── evaluate_model_accuracy.py # Script to evaluate model performance
├── utils/
│   ├── preprocess.py              # Preprocessing utilities
│   └── environment.md             # Setup and environment details
├── model_architecture.png         # Diagram of model architecture
├── model_report.xlsx              # Performance and metrics report
├── INTEL_Project_Documentation.pdf  # Full project documentation
├── problem_scope_document.md      # Problem statement and scope
├── Working Methodology of the SQLI_XSS Detector Model.mp4
├── README.md
├── requirements.txt               # Python dependencies
└── .                              # Project root


Getting Started:
rerequisites
Python 3.11 (Recommended)
pip package manager
Installation
Clone the repository and install dependencies:

git clone https://github.com/saivarma-2367/ai_sql_xss_detector
cd AI_SQL_XSS_DETECTOR
pip install -r requirements.txt


Environment Setup:
For complete environment setup and package versions, refer to:
utils/environment.md


Key Components:
1. Data
data/raw_dataset.csv: Original dataset with mixed entries.
data/filtered_SQL_XSS_only.csv: Filtered dataset with SQLi and XSS samples.
data/test_input.xlsx: Example inputs for testing the model.

2. Machine Learning Model
ml_model/train.py: Script to train the detection model.
ml_model/rf_model_compressed.pkl: Compressed and trained Random Forest model.
ml_model/vectorizer.pkl: Vectorizer (e.g., TF-IDF) for text preprocessing.
ml_model/label_encoder.pkl: Encodes the output labels (SQLi, XSS, benign, etc.).

3. API (Streamlit Web App)
A user-friendly interface is built using Streamlit:
To launch the app:
bash
streamlit run api/app.py

4. Model Evaluation
To evaluate the accuracy of the trained model:
bash
python test/evaluate_model_accuracy.py

6. Utilities
utils/preprocess.py: Functions for cleaning and preparing input data.
utils/environment.md: Full environment setup guide.


Documentation and Reports:
File	Description
INTEL_Project_Documentation.pdf	                                       Full documentation of the AI SQL/XSS Detector
model_report.xlsx                                                      Evaluation metrics, accuracy, precision, recall, etc.
problem_scope_document.md	                                             Overview of the problem and project scope
model_architecture.png	                                               Diagram of the model pipeline/structure
Working Methodology of the SQLI_XSS Detector Modelmp4	                 Project demonstration and usage guide (video)
