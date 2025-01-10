import streamlit as st
from model import load_model, predict_image
from PIL import Image

# Load the trained model
model = load_model('trained_net.pth')

# Streamlit app title and description
st.title("CIFAR-10 Dataset Classification")
st.write("Upload an image to get the predicted class using the trained model.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file).convert('RGB')

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict the class using the model
    st.write("Classifying...")
    prediction = predict_image(image, model)

    # Display the prediction result
    st.success(f"{prediction}")
