# Mango Classifier

A Streamlit-based UI for testing and using a mango classification model.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your trained model file (e.g., `model.pth`) in the project directory

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. Load your trained model using the "Load Model" button in the sidebar
2. Upload an image of a mango
3. Click "Classify" to get the prediction

The app will display:
- The uploaded image
- The predicted class
- The probabilities for each class
