# Gender Classification Model - Streamlit Deployment

A web application for gender classification using a pre-trained Keras model deployed on Streamlit Cloud.

## Features

- 📸 Upload images for gender classification
- 🎯 Real-time predictions with confidence scores
- 💻 User-friendly web interface
- ☁️ Easy deployment to Streamlit Cloud

## Files Included

- `app.py` - Main Streamlit application
- `Gender_classification.keras` - Pre-trained Keras model
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit configuration

## Local Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gender_classification.git
cd gender_classification
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app locally:
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Deployment to Streamlit Cloud

### Step 1: Push to GitHub
1. Create a new repository on GitHub
2. Push your files:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/gender_classification.git
git push -u origin main
```

### Step 2: Deploy to Streamlit Cloud
1. Go to [Streamlit Cloud](https://streamlit.io/cloud)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository, branch, and main file (`app.py`)
5. Click "Deploy"

Your app will be live at: `https://<your-username>-gender-classification.streamlit.app/`

## Usage

1. Open the web app
2. Upload an image (JPG or PNG)
3. Wait for the model to process
4. View the prediction and confidence score

### Tips for Best Results
- Use clear, well-lit photos
- Front-facing images work best
- Avoid heavy shadows
- Ensure the face is clearly visible
- High resolution images are preferred

## Model Information

- **Type:** Convolutional Neural Network (CNN)
- **Classes:** Male, Female
- **Input Size:** 224x224 pixels
- **Output:** Class prediction with confidence score

## License

This project is open source and available under the MIT License.

## Troubleshooting

### Model not found error
- Ensure `Gender_classification.keras` is in the same directory as `app.py`
- Check file name spelling and case sensitivity

### Deployment fails
- Verify all files are in the GitHub repository
- Check that `requirements.txt` includes all necessary packages
- Review Streamlit Cloud deployment logs

## Contact

For issues or questions, please open an issue on the GitHub repository.
