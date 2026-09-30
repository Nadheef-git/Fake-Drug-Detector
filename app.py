import streamlit as st
import numpy as np
import joblib
from PIL import Image
import cv2
import os

# Load the trained model once, when the app starts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'medicine_classifier_rf.pkl')
rf_model = joblib.load(model_path)

def extract_features(img_array):
    img_uint8 = (img_array * 255).astype('uint8')
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    r_mean, g_mean, b_mean = img_array[:,:,0].mean(), img_array[:,:,1].mean(), img_array[:,:,2].mean()
    r_std, g_std, b_std = img_array[:,:,0].std(), img_array[:,:,1].std(), img_array[:,:,2].std()
    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.mean() / 255.0
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return np.array([[r_mean, g_mean, b_mean, r_std, g_std, b_std, edge_density, sharpness]])

st.title('Medicine Packaging Authenticity Checker')
st.warning('⚠️ Proof-of-concept only — not validated for real-world counterfeit detection. See project limitations.')

uploaded_file = st.file_uploader('Upload a photo of the medicine packaging', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB').resize((128, 128))
    img_array = np.array(img) / 255.0

    st.image(img, caption='Uploaded Image', width=300)

    features = extract_features(img_array)
    prediction = rf_model.predict(features)[0]
    probability = rf_model.predict_proba(features)[0]

    label = 'Real' if prediction == 1 else 'Fake'
    confidence = probability[prediction]

    st.write(f'**Prediction: {label}**')
    st.write(f'Confidence: {confidence:.1%}')

