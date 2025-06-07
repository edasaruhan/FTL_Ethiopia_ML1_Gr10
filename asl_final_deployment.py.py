import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
import joblib
from PIL import Image

# Load the model
model = load_model('asl_cnn_model.h5')

# Mapping from class index to words
class_mapping = {
    0: "Hello",
    1: "Thank You",
    2: "Please",
    # Add more mappings as needed
}

# Streamlit app
st.set_page_config(page_title="ASL Translator", layout="wide")
st.title("AI Powered Sign Language Translator")
st.write("Upload an image of a sign to get the corresponding word.")

# Sidebar for additional options
with st.sidebar:
    st.header("Instructions")
    st.write("1. Upload an image of a sign.")
    st.write("2. Click 'Predict' to see the translation.")
    st.write("3. Ensure the image is clear and well-lit.")

# Image upload
uploaded_image = st.file_uploader("Upload an image of a sign", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # BGR format

    # Display as RGB
    st.image(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB), caption='Uploaded Image', use_column_width=True)

    # Preprocess for prediction (keep as BGR if model was trained on BGR)
    image_resized = cv2.resize(image_cv, (128, 128))
    image_array = image_resized / 255.0
    image_array = image_array.reshape(1, 128, 128, 3)

    if st.button("Predict"):
        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        label_encoder = joblib.load('label_encoder.pkl')
        predicted_letter = label_encoder.inverse_transform([predicted_class_index])[0]
        
        label_to_amharic = {'che': 'ቸ','gne': 'ኘ', 'ha':'ሀ', 'he':'ሕ', 'hhe':'ኸ', 'ke':'ከ', 'le':'ለ', 'me':'መ', 'ne':'ነ', 'qe':'ቀ'}

        amharic_letter = label_to_amharic.get(predicted_letter, 'Unknown')

        st.write("### Predicted Sign Language Word:")
        st.success(amharic_letter)



    # Add an expander for more information
    with st.expander("About the Model"):
        st.write("This model translates American Sign Language signs into words.")
        st.write("Make sure to upload clear images for accurate predictions.")