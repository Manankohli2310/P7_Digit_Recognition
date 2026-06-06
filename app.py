import streamlit as st
import numpy as np
import joblib

from PIL import Image
from streamlit_drawable_canvas import st_canvas

# =====================================
# Page Configuration
# =====================================

st.set_page_config(
    page_title="Digit Recognition",
    page_icon="✍️",
    layout="centered"
)

# =====================================
# Load Model and Scaler
# =====================================

@st.cache_resource
def load_model():
    model = joblib.load("model/svm_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# =====================================
# Title
# =====================================

st.title("✍️ Digit Recognition")
st.markdown(
    """
    Draw a digit (0–9) inside the canvas and click **Predict Digit**.
    The trained SVM model will recognize your handwritten number.
    """
)

st.divider()

# =====================================
# Drawing Canvas
# =====================================

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# =====================================
# Prediction Function
# =====================================

def preprocess_image(image_array):

    from PIL import Image
    import numpy as np

    # RGBA -> Grayscale
    img = Image.fromarray(
        image_array.astype("uint8")
    ).convert("L")

    img_array = np.array(img)

    # Binary Threshold
    img_array = np.where(
        img_array > 50,
        255,
        0
    )

    # Find Digit Area
    coords = np.argwhere(img_array > 0)

    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    digit = img_array[
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    # Resize digit to 20x20
    digit_img = Image.fromarray(
        digit.astype(np.uint8)
    )

    digit_img = digit_img.resize(
        (20, 20)
    )

    digit_array = np.array(digit_img)

    # Create 28x28 canvas
    mnist_canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    mnist_canvas[
        4:24,
        4:24
    ] = digit_array

    final_image = mnist_canvas.reshape(
        1,
        784
    )

    return final_image

# =====================================
# Predict Button
# =====================================

if st.button("🔍 Predict Digit"):

    if canvas_result.image_data is None:

        st.warning("Please draw a digit first.")

    else:

        image = canvas_result.image_data

        processed_image = preprocess_image(image)

        if processed_image is None:
            st.warning(
                "Please draw a digit."
            )
            st.stop()

        scaled_image = scaler.transform(
            processed_image
        )
        st.image(
        processed_image.reshape(28,28),
        caption="Processed MNIST Image"
        )

        prediction = model.predict(scaled_image)[0]

        try:
            probabilities = model.predict_proba(scaled_image)[0]
            confidence = np.max(probabilities) * 100

            st.success(
                f"Predicted Digit: {prediction}"
            )

            st.info(
                f"Confidence: {confidence:.2f}%"
            )

        except:

            st.success(
                f"Predicted Digit: {prediction}"
            )

# =====================================
# Clear Instructions
# =====================================

st.divider()

st.subheader("Instructions")

st.markdown(
    """
    - Draw only one digit at a time.
    - Draw near the center of the canvas.
    - Use thick strokes.
    - Digits 0–9 are supported.
    - Click **Predict Digit** after drawing.
    """
)