## Problem Scope Document

### Project Title
AI/ML-Powered SQL Injection & XSS Attack Detection

### Problem Statement
Modern web applications are vulnerable to injection-based attacks. Manual rule-based systems are often ineffective against evolving attack patterns.

### Objective
Build two AI models (ML and DL):
1. To detect and prevent SQL Injection attacks
2. To detect and prevent Cross-site Scripting (XSS)

### Goals
- Train supervised classifiers on labeled data
- Evaluate models with precision, recall, accuracy
- Integrate blocking logic into an API later

### Dataset
- Source: Kaggle SQLi-XSS Dataset
- Labels: SQLInjection, XSS, Normal

### Tools
- Python, Scikit-learn, PyTorch/TensorFlow, Wireshark (for traffic simulation)

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score