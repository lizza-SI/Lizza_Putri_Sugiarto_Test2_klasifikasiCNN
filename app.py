
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input
)

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Oceanic Life CNN",
    page_icon="🌊",
    layout="wide"
)

# =====================================================
# 2. MODEL CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "output_final"
    / "models"
    / "mobilenetv2_stage2_best.keras"
)

IMG_SIZE = (224, 224)
# Batas confidence awal (perlu validasi lebih lanjut)
CONFIDENCE_THRESHOLD = 0.60

CLASS_NAMES = [
    "Coral",
    "Crustaceans",
    "Fish",
    "Octopuses",
    "Other marine organisms",
    "Shark",
    "Sponges"
]

# =====================================================
# 3. LOAD MODEL
# =====================================================

@st.cache_resource
def load_cnn_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    return model


# =====================================================
# 4. IMAGE PREPROCESSING
# =====================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = preprocess_input(
        image_array
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =====================================================
# 5. PREDICTION
# =====================================================

def predict_image(model, image):

    processed_image = preprocess_image(
        image
    )

    predictions = model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    return (
        predicted_class,
        confidence,
        predictions
    )


# =====================================================
# 6. DASHBOARD HEADER
# =====================================================

st.title("🌊 Oceanic Life Classification")
st.subheader(
    "Marine Organism Classification Using "
    "Convolutional Neural Network"
)

st.write(
    "Aplikasi klasifikasi citra organisme laut "
    "menggunakan arsitektur MobileNetV2 "
    "dengan transfer learning dan fine-tuning."
)

st.divider()

# =====================================================
# 7. SIDEBAR
# =====================================================

st.sidebar.title("About Project")

st.sidebar.write(
    "Oceanic Life CNN merupakan proyek "
    "klasifikasi citra organisme laut "
    "menggunakan deep learning."
)

st.sidebar.markdown("### Model Information")

st.sidebar.write("Architecture: MobileNetV2")
st.sidebar.write("Input: 224 × 224 × 3")
st.sidebar.write("Output classes: 7")
st.sidebar.write("Training: Stage 1 + Stage 2")

# =====================================================
# 8. LOAD MODEL
# =====================================================

try:

    model = load_cnn_model()

    st.sidebar.success(
        "CNN Model Loaded Successfully"
    )

except Exception as error:

    st.error(
        f"Gagal memuat model: {error}"
    )

    st.stop()



# =====================================================
# 9. IMAGE UPLOAD
# =====================================================

st.header("Image Classification")

uploaded_file = st.file_uploader(
    "Upload gambar organisme laut",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        image = Image.open(
            uploaded_file
        ).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Uploaded Image")

            st.image(
                image,
                use_container_width=True
            )

        # =============================================
        # 10. CLASSIFICATION
        # =============================================

        with col2:

            st.subheader("Classification Result")

            with st.spinner("Analyzing image..."):

                predicted_class, confidence, predictions = (
                    predict_image(model, image)
                )

            if confidence >= CONFIDENCE_THRESHOLD:

                st.success(
                    f"Predicted Class: {predicted_class}"
                )

                prediction_status = "confident"

            else:

                st.warning(
                    "Prediksi belum cukup meyakinkan."
                )

                st.info(
                    "Gambar mungkin berada di luar kategori "
                    "organisme laut atau memiliki karakteristik "
                    "visual yang sulit dikenali model."
                )

                prediction_status = "uncertain"

            st.metric(
                "Highest Class Probability",
                f"{confidence * 100:.2f}%"
            )

            st.progress(confidence)

        # =============================================
        # 11. CLASS PROBABILITIES
        # =============================================

        st.divider()

        st.subheader("Prediction Probability")

        probability_data = sorted(
            zip(CLASS_NAMES, predictions * 100),
            key=lambda x: x[1],
            reverse=True
        )

        for class_name, probability in probability_data:

            st.write(
                f"**{class_name}** — {probability:.2f}%"
            )

            st.progress(
                float(probability / 100)
            )

        # =============================================
        # 12. PREDICTION ANALYSIS
        # =============================================

        st.divider()

        st.subheader("Prediction Analysis")

        if prediction_status == "confident":

            st.write(
                f"Model memprediksi gambar sebagai "
                f"**{predicted_class}** dengan probabilitas "
                f"**{confidence * 100:.2f}%**."
            )

        else:

            st.write(
                "Model belum menghasilkan prediksi "
                "yang cukup meyakinkan."
            )

            st.write(
                "Periksa kembali gambar dan pastikan "
                "objek termasuk dalam kategori organisme laut."
            )

        st.caption(
            "Confidence merupakan probabilitas keluaran model, "
            "bukan jaminan bahwa prediksi benar. Threshold "
            "digunakan sebagai penyaring awal dan belum "
            "merupakan deteksi OOD yang tervalidasi."
        )

    except Exception as error:

        st.error(
            f"Gagal memproses gambar: {error}"
        )

else:

    st.info(
        "Silakan upload gambar organisme laut "
        "untuk memulai proses klasifikasi."
    )


# =====================================================
# 13. FOOTER
# =====================================================

st.divider()

st.caption(
    "Oceanic Life CNN | MobileNetV2 "
    "| Image Classification Project"
)