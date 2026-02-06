
import streamlit as st
import numpy as np
import tensorflow as tf
import pickle

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the Keras model
model = tf.keras.models.load_model('model.keras')

# Define class names
class_names = ['Setosa', 'Versicolor', 'Virginica']

# Streamlit app title and header
st.title('Iris Flower Classification')
st.header('Predict the species of Iris flower based on its features')

# Input widgets for features
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.4, 0.1)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.4, 0.1)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 1.3, 0.1)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 0.2, 0.1)

# Predict button
if st.button('Predict'):
    # Collect input values into a NumPy array
    input_features = np.array([
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]).reshape(1, -1)

    # Scale the input features
    scaled_features = scaler.transform(input_features)

    # Make prediction
    predictions = model.predict(scaled_features)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    prediction_probability = predictions[0][predicted_class_index]

    # Display results
    st.success(f'Predicted Iris Species: **{predicted_class_name}**')
    st.write(f'Prediction Probability: **{prediction_probability:.2f}**')

