# app.py
import streamlit as st
import pandas as pd
import joblib
import io

st.title("🛡️ AI Network Security - Classifier")

# Load compressed ML components
model = joblib.load('ml_model/rf_model_compressed.pkl')
vectorizer = joblib.load('ml_model/vectorizer.pkl')
encoder = joblib.load('ml_model/label_encoder.pkl')

# Single input
st.header("🔠 Single Sentence or URL")
text = st.text_area("Enter input:")
if text:
    vec = vectorizer.transform([text])
    pred = model.predict(vec)
    label = encoder.inverse_transform(pred)[0]
    st.success(f"🔍 Prediction: `{label}`")

# Excel upload
st.header("📁 Upload Excel (.xlsx) with 'input' column")
file = st.file_uploader("Upload file", type=['xlsx'])

if file:
    try:
        df = pd.read_excel(file)
        if 'input' not in df.columns:
            st.error("Column `input` not found.")
        else:
            vec = vectorizer.transform(df['input'].astype(str))
            df['Prediction'] = encoder.inverse_transform(model.predict(vec))
            st.dataframe(df[['input', 'Prediction']])

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False)
            output.seek(0)

            st.download_button("📥 Download Excel", output, "predictions.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as e:
        st.error(f"Error: {e}")
