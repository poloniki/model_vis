import streamlit as st
import tensorflow as tf
import tensorspace as ts

# Define the path to the TensorFlow.js model
TFJS_MODEL_PATH = 'my_conv_model_tfjs/model.json'

# Load the TensorFlow.js model
model = tf.keras.models.load_model(TFJS_MODEL_PATH)

# Convert the model to TensorSpace.js format
converted_model = ts.Model()
converted_model.load(TFJS_MODEL_PATH, 'tfjs')

# Create the Streamlit app
st.title('TensorSpace.js Model Visualization')

# Display the model
container = st.empty()
visualizer = ts.LayerVisualizer(converted_model, container)
visualizer.init()
