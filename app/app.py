import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# Title
st.set_page_config(page_title="🔥BlazeTrack", layout="centered")
st.title("🔥 Fire or Non-Fire Detection")
st.write("Upload an image to check if it contains fire. Choose a model and view Grad-CAM heatmap.")

# Load models
@st.cache_resource
def load_models():
    cnn_model = tf.keras.models.load_model("fire_detection_model.keras")
    mobilenet_model = tf.keras.models.load_model("transfer_model.keras")
    return cnn_model, mobilenet_model

cnn_model, mobilenet_model = load_models()

# Prediction function
def predict(model, img_array):
    preds = model.predict(img_array)[0]
    class_names = ["Non-Fire", "Fire"]
    label = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))
    return label, confidence, preds

# Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    return heatmap

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Model selection
model_choice = st.radio("Choose a model:", ("CNN", "MobileNetV2"))

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = image.resize((128, 128))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    # Predict
    model = cnn_model if model_choice == "CNN" else mobilenet_model
    label, confidence, preds = predict(model, img_array)

    st.markdown(f"### Prediction: `{label}`")
    st.progress(int(confidence * 100))
    st.write(f"Confidence Score: `{confidence:.2f}`")

    # Grad-CAM
    st.subheader("Grad-CAM Heatmap")
    heatmap = make_gradcam_heatmap(img_array, model)
    heatmap = cv2.resize(heatmap, (image.size))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(np.array(image), 0.6, heatmap, 0.4, 0)

    st.image(superimposed_img, caption="Grad-CAM", use_column_width=True)
