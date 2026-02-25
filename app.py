import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os

# Set page config
st.set_page_config(
    page_title="Gender Classification",
    page_icon="👤",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #2E86AB;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the pre-trained gender classification model"""
    model_path = "Gender_classification.keras"
    
    try:
        if os.path.exists(model_path):
            # Try loading with safe_mode=False for compatibility with older saved models
            try:
                model = keras.models.load_model(model_path, safe_mode=False)
            except TypeError:
                # If safe_mode is not supported, try standard loading
                model = keras.models.load_model(model_path)
            return model
        else:
            st.error(f"❌ Model file not found at: {os.path.abspath(model_path)}")
            st.info("**Solution:** Please ensure 'Gender_classification.keras' is:")
            st.markdown("""
            - In the same directory as `app.py`
            - Committed and pushed to your GitHub repository
            - Not in .gitignore
            """)
            st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.warning("**The model file appears to be corrupted or from an incompatible TensorFlow version.**")
        st.info("""
        **To fix this:**
        1. Go back to where you trained the model
        2. Re-save it using the latest code:
        ```python
        model.save('Gender_classification.keras')
        ```
        3. Replace the corrupted file in your repository
        """)
        st.stop()

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    img = image.resize(target_size)
    img_array = np.array(img)
    
    # Normalize the image
    if img_array.max() > 1:
        img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def main():
    st.markdown('<div class="main-title">👤 Gender Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload an image to classify gender</div>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar information
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. Upload an image (JPG, PNG)
        2. The model will process the image
        3. View the prediction result
        
        **Supported formats:** JPG, PNG
        
        **Note:** Best results with clear, front-facing photos
        """)
        
        st.header("ℹ️ Model Info")
        st.info(f"**Model:** Gender Classification CNN\n\n**Classes:** Male, Female")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear photo for best results"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(image, use_column_width=True)
        
        # Make prediction
        with st.spinner("🔍 Analyzing image..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image, verbose=0)
            
            # Get class and confidence
            confidence = float(prediction[0][0]) * 100 if len(prediction[0]) == 1 else max(prediction[0]) * 100
            
            # Determine gender class (assuming binary classification)
            if len(prediction[0]) == 1:
                # Sigmoid output
                predicted_class = "Female" if prediction[0][0] < 0.5 else "Male"
                confidence = (1 - prediction[0][0]) * 100 if prediction[0][0] < 0.5 else prediction[0][0] * 100
            else:
                # Softmax output
                class_idx = np.argmax(prediction[0])
                predicted_class = "Male" if class_idx == 0 else "Female"
                confidence = prediction[0][class_idx] * 100
        
        with col2:
            st.subheader("🎯 Prediction Result")
            
            # Display result with color coding
            if predicted_class == "Male":
                st.markdown(f'<h2 style="color: #3498db; text-align: center;">👨 {predicted_class}</h2>', unsafe_allow_html=True)
            else:
                st.markdown(f'<h2 style="color: #e74c3c; text-align: center;">👩 {predicted_class}</h2>', unsafe_allow_html=True)
            
            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%",
                delta=None
            )
            
            # Confidence bar
            st.progress(min(confidence / 100, 1.0))
            
            if confidence < 70:
                st.warning("⚠️ Low confidence. Result may be unreliable.")
            elif confidence >= 90:
                st.success("✅ High confidence prediction")
    
    else:
        st.info("👆 Please upload an image to get started!")
        
        # Add sample instructions
        with st.expander("💡 Tips for best results"):
            st.markdown("""
            - Use clear, well-lit photos
            - Front-facing images work best
            - Avoid heavy shadows or obstructions
            - Ensure the face is clearly visible
            - High resolution images are preferred
            """)

if __name__ == "__main__":
    main()
