# train.py
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocess import load_data

# Load and preprocess
X_train, X_test, y_train, y_test, encoder = load_data('data/filtered_SQL_XSS_only.csv')

# Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Measure latency (average per prediction)
start_time = time.time()
y_pred = model.predict(X_test_vec)
end_time = time.time()

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
total_time = end_time - start_time
latency_per_sample = total_time / X_test_vec.shape[0]


# Output
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print(f"⏱️ Average Latency per Prediction: {latency_per_sample * 1000:.4f} ms")

# Save with compression
joblib.dump(model, 'ml_model/rf_model_compressed.pkl', compress=3)
joblib.dump(vectorizer, 'ml_model/vectorizer.pkl', compress=3)
joblib.dump(encoder, 'ml_model/label_encoder.pkl', compress=3)
