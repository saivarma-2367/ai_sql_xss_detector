import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_excel('data/test_input.xlsx')

model = joblib.load('ml_model/rf_model_compressed.pkl')
vectorizer = joblib.load('ml_model/vectorizer.pkl')
encoder = joblib.load('ml_model/label_encoder.pkl')

print("LabelEncoder classes:", encoder.classes_)

X_vec = vectorizer.transform(df['input'].astype(str))
y_pred = model.predict(X_vec)
y_true = encoder.transform(df['actual'].astype(str)) 

df['Predicted'] = encoder.inverse_transform(y_pred)

accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=encoder.classes_, output_dict=True)

print(f"\nAccuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=encoder.classes_))

report_df = pd.DataFrame(report).transpose()
with pd.ExcelWriter("model_report.xlsx") as writer:
    df.to_excel(writer, index=False, sheet_name="Predictions")
    report_df.to_excel(writer, sheet_name="Metrics")
