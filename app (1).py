
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="AI Image Colorization",
    page_icon="🎨",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "image_colorization_model (1).keras"
    )

model = load_model()

# Function to convert image into PNG bytes
def convert_to_png(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# Application title
st.title("🎨 AI-Based Image Colorization")

st.write(
    "Convert black-and-white images into colored images using Deep Learning."
)

# Upload image
uploaded_file = st.file_uploader(
    "Upload a black-and-white image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        # Read uploaded image
        image = Image.open(uploaded_file).convert("RGB")

        # Display original image
        st.subheader("Original Image")

        original_bytes = convert_to_png(image)

        st.image(
            original_bytes,
            caption="Original Image",
            use_container_width=True
        )

        # Colorization button
        if st.button("🎨 Colorize Image", type="primary"):

            with st.spinner("Generating colorized image..."):

                # Preprocessing
                img = np.array(image)

                img = cv2.resize(
                    img,
                    (128, 128)
                )

                gray = cv2.cvtColor(
                    img,
                    cv2.COLOR_RGB2GRAY
                )

                X = gray.astype("float32") / 255.0

                X = X.reshape(
                    1, 128, 128, 1
                )

                # Model prediction
                prediction = model.predict(
                    X,
                    verbose=0
                )

                # Create LAB image
                lab = np.zeros(
                    (128, 128, 3),
                    dtype=np.uint8
                )

                lab[:, :, 0] = gray

                lab[:, :, 1:] = np.clip(
                    prediction[0] * 128 + 128,
                    0,
                    255
                ).astype(np.uint8)

                # Convert LAB to RGB
                result = cv2.cvtColor(
                    lab,
                    cv2.COLOR_LAB2RGB
                )

                # Convert result to PIL image
                colorized_image = Image.fromarray(
                    result.astype(np.uint8)
                )

                # Display colorized image
                st.subheader("Colorized Image")

                colorized_bytes = convert_to_png(
                    colorized_image
                )

                st.image(
                    colorized_bytes,
                    caption="AI Generated Color Image",
                    use_container_width=True
                )

                # Download button
                st.download_button(
                    label="⬇️ Download Colorized Image",
                    data=colorized_bytes,
                    file_name="colorized_image.png",
                    mime="image/png"
                )

    except Exception as e:
        st.error("Something went wrong while processing the image.")
        st.exception(e)

st.divider()

st.caption(
    "Developed using CNN and Deep Learning"
)
