import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the model
model = load_model('model/asl_cnn_model.h5')

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
    # Open and process the image
    image = Image.open(uploaded_image)
    
    # Convert to RGB format
    image = image.convert("RGB")
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Resize the image
    image = image.resize((224, 224))  # Adjust size based on model training
    image_array = np.array(image) / 255.0  # Normalize if needed
    image_array = image_array.reshape(1, 224, 224, 3)  # Adjust shape

    # Predict button
    if st.button("Predict"):
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions)  # Get the class index
        predicted_word = class_mapping.get(predicted_class, "Unknown Sign")
        st.write("### Predicted Sign Language Word:")
        st.success(predicted_word)

    # Add an expander for more information
    with st.expander("About the Model"):
        st.write("This model translates American Sign Language signs into words.")
        st.write("Make sure to upload clear images for accurate predictions.")