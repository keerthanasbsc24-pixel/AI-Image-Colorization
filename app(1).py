
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(
    page_title="AI Image Colorization",
    page_icon="🎨",
    layout="wide"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("image_colorization_model.keras")

model = load_model()

st.title("🎨 AI-Based Image Colorization")
st.write("Convert black-and-white images into colored images using Deep Learning.")

uploaded_file = st.file_uploader(
    "Upload a black-and-white image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, width=400)

    if st.button("Colorize Image", type="primary"):

        with st.spinner("Generating colorized image..."):

            img = np.array(image)
            img = cv2.resize(img, (128, 128))

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            X = gray.astype("float32") / 255.0
            X = X.reshape(1, 128, 128, 1)

            prediction = model.predict(X, verbose=0)

            lab = np.zeros((128, 128, 3), dtype=np.uint8)
            lab[:, :, 0] = gray

            lab[:, :, 1:] = np.clip(
                prediction[0] * 128 + 128,
                0, 255
            ).astype(np.uint8)

            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

            st.subheader("Colorized Image")
            st.image(result, width=400)

            buffer = io.BytesIO()
            Image.fromarray(result).save(buffer, format="PNG")

            st.download_button(
                "Download Colorized Image",
                data=buffer.getvalue(),
                file_name="colorized_image.png",
                mime="image/png"
            )

st.divider()
st.caption("Developed using CNN and Deep Learning")
