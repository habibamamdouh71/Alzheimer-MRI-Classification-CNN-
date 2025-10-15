import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="Alzheimer's MRI Classification",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🧠 Alzheimer MRI Classification</div>', unsafe_allow_html=True)
st.subheader("📁 Upload an MRI scan")

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    "This app uses a deep learning model to classify Alzheimer progression stages from MRI scans."
)
st.sidebar.title("Model Information")
st.sidebar.write("**Classes:** Mild Demented, Moderate Demented, Non Demented, Very Mild Demented")
st.sidebar.write("**Input size:** 128x128 pixels")
st.sidebar.write("**Model:** Custom CNN")

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("alzheimer_model.keras")
    return model

# Preprocess the image
def preprocess_image(image):
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    # Convert grayscale to RGB if needed
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Class names for Alzheimer's
class_names = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Scan", width=300)
    processed_image = preprocess_image(image)

    if st.button("Classify Image", type="primary"):
        with st.spinner("Analyzing the scan..."):
            model = load_model()
            predictions = model.predict(processed_image)

            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = np.max(predictions[0])
            all_probs = predictions[0]

            # Results
            st.subheader("Alzheimer's Classification Result")
            
            # Show main prediction with color coding
            if predicted_class == "Non Demented":
                st.success(f"### ✅ {predicted_class} (Confidence: {confidence:.2%})")
            elif predicted_class == "Very Mild Demented":
                st.warning(f"### ⚠️ {predicted_class} (Confidence: {confidence:.2%})")
            elif predicted_class == "Mild Demented":
                st.warning(f"### ⚠️ {predicted_class} (Confidence: {confidence:.2%})")
            else:  # Moderate Demented
                st.error(f"### ❌ {predicted_class} (Confidence: {confidence:.2%})")

            # Show all probabilities
            st.subheader("Confidence Levels for All Classes:")
            for i, class_name in enumerate(class_names):
                prob = all_probs[i]
                st.write(f"{class_name}: {prob:.2%}")

# Instructions when no file is uploaded
else:
    st.info("👆 Please upload an MRI scan to get started")