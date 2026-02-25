import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import os
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Gender Classification Model",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================
st.markdown("""
    <style>
    .main-container {
        padding: 2rem;
    }
    .title {
        text-align: center;
        color: #1f77b4;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHE MODEL LOADING
# ============================================================================
@st.cache_resource
def load_model():
    """Load the pre-trained gender classification model"""
    model_path = "Gender_classification.keras"
    
    try:
        if not os.path.exists(model_path):
            return None, "Model file not found"
        
        # Try loading with different methods
        try:
            model = keras.models.load_model(model_path, safe_mode=False)
        except TypeError:
            # Fallback for TensorFlow versions without safe_mode parameter
            model = keras.models.load_model(model_path)
        
        return model, "Success"
    
    except Exception as e:
        return None, str(e)

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        img = image.resize(target_size)
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize the image
        if img_array.max() > 1:
            img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, None
    
    except Exception as e:
        return None, f"Error preprocessing image: {str(e)}"

# ============================================================================
# MAKE PREDICTION
# ============================================================================
def make_prediction(model, processed_image):
    """Make gender prediction on the processed image"""
    try:
        prediction = model.predict(processed_image, verbose=0)
        
        # Handle different output shapes
        if len(prediction.shape) == 2:
            if prediction.shape[1] == 1:
                # Binary classification with sigmoid
                confidence = float(prediction[0][0])
                predicted_class = "Female" if confidence < 0.5 else "Male"
                confidence_score = (1 - confidence) * 100 if confidence < 0.5 else confidence * 100
            else:
                # Multi-class classification with softmax
                class_idx = np.argmax(prediction[0])
                confidence_score = float(np.max(prediction[0])) * 100
                classes = ["Female", "Male"]
                predicted_class = classes[class_idx] if class_idx < len(classes) else f"Class {class_idx}"
        else:
            return None, None, "Unexpected model output shape"
        
        return predicted_class, confidence_score, None
    
    except Exception as e:
        return None, None, f"Error making prediction: {str(e)}"

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Title
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="title">👤 Gender Classification</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle">AI-Powered Image Analysis</div>', unsafe_allow_html=True)
    
    # Load model
    model, load_status = load_model()
    
    if model is None:
        st.error("❌ Failed to load model")
        st.error(f"Error: {load_status}")
        st.stop()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🖼️ Predict", "📋 Instructions", "ℹ️ About"])
    
    # ========================================================================
    # TAB 1: PREDICTION
    # ========================================================================
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Upload Image")
            uploaded_file = st.file_uploader(
                "Choose an image (JPG, PNG, JPEG)",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear photo of a face for best results"
            )
            
            if uploaded_file is not None:
                # Load and display image
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True, caption="Uploaded Image")
        
        with col2:
            st.subheader("🎯 Prediction Result")
            
            if uploaded_file is not None:
                with st.spinner("🔍 Analyzing image..."):
                    # Preprocess image
                    processed_image, prep_error = preprocess_image(image)
                    
                    if prep_error:
                        st.error(prep_error)
                    else:
                        # Make prediction
                        predicted_class, confidence_score, pred_error = make_prediction(model, processed_image)
                        
                        if pred_error:
                            st.error(pred_error)
                        else:
                            # Display results
                            col_res1, col_res2 = st.columns(2)
                            
                            with col_res1:
                                if predicted_class == "Male":
                                    st.markdown(
                                        f'<div class="result-box success-box"><h2 style="margin:0; color:#155724;">👨 {predicted_class}</h2></div>',
                                        unsafe_allow_html=True
                                    )
                                else:
                                    st.markdown(
                                        f'<div class="result-box success-box"><h2 style="margin:0; color:#155724;">👩 {predicted_class}</h2></div>',
                                        unsafe_allow_html=True
                                    )
                            
                            with col_res2:
                                st.metric(
                                    label="Confidence Score",
                                    value=f"{confidence_score:.1f}%",
                                    delta=None
                                )
                            
                            # Confidence bar
                            st.progress(min(confidence_score / 100, 1.0))
                            
                            # Confidence level indicator
                            if confidence_score >= 90:
                                st.success("✅ Very High Confidence")
                            elif confidence_score >= 75:
                                st.info("✓ High Confidence")
                            elif confidence_score >= 60:
                                st.warning("⚠️ Medium Confidence")
                            else:
                                st.error("❌ Low Confidence - Result may be unreliable")
                            
                            # Timestamp
                            st.caption(f"Prediction made at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            else:
                st.info("👆 Upload an image to see predictions")
    
    # ========================================================================
    # TAB 2: INSTRUCTIONS
    # ========================================================================
    with tab2:
        st.subheader("📋 How to Use")
        
        st.markdown("""
        ### Step-by-Step Guide:
        
        1. **Upload an Image**
           - Click the upload area in the "Predict" tab
           - Select a JPG or PNG image file
           - The image should contain a clear face
        
        2. **Wait for Analysis**
           - The model will process your image
           - This usually takes a few seconds
        
        3. **View Results**
           - See the predicted gender (Male/Female)
           - Check the confidence score (0-100%)
           - Higher confidence = more reliable prediction
        
        ### Tips for Best Results:
        - ✅ Use clear, well-lit photos
        - ✅ Face should be front-facing
        - ✅ High resolution images work better
        - ✅ Avoid heavy shadows or obstructions
        - ✅ Ensure face is clearly visible
        - ❌ Avoid heavily edited or filtered images
        - ❌ Don't use cartoons or drawings
        
        ### Confidence Score Guide:
        - **90-100%**: Very High Confidence ✅
        - **75-89%**: High Confidence ✓
        - **60-74%**: Medium Confidence ⚠️
        - **Below 60%**: Low Confidence - Unreliable ❌
        """)
    
    # ========================================================================
    # TAB 3: ABOUT
    # ========================================================================
    with tab3:
        st.subheader("ℹ️ About This Application")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Model Information
            - **Type**: Convolutional Neural Network (CNN)
            - **Task**: Binary Gender Classification
            - **Classes**: Male, Female
            - **Input Size**: 224×224 pixels
            - **Architecture**: Deep Learning Model
            """)
        
        with col2:
            st.markdown("""
            ### Application Info
            - **Framework**: Streamlit
            - **Backend**: TensorFlow/Keras
            - **Language**: Python
            - **Deployment**: Streamlit Cloud
            - **Version**: 1.0
            """)
        
        st.divider()
        
        st.markdown("""
        ### Disclaimer
        This model provides predictions based on visual features in images. 
        Results should not be used for critical decision-making without human verification. 
        The model may have limitations and biases.
        
        ### Privacy
        - Images are processed locally
        - No images are stored or logged
        - No data collection occurs
        """)
        
        st.markdown("""
        ---
        **Created with ❤️ using Streamlit & TensorFlow**
        """)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
