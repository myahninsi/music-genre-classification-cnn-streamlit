import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tempfile
import os
import joblib
import pandas as pd
import plotly.express as px
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pydub import AudioSegment

# ====== PAGE SETUP ======
st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵", layout="centered")
st.title("🎵 Music Genre Classifier")
st.markdown("""
### 🚀 How It Works

➡️ **Step 1**: Select a model from the dropdown (CNN or ML)  
➡️ **Step 2**: Upload your `.mp3` or `.wav` audio file  
➡️ **Step 3**: We generate a Mel spectrogram from the audio  
➡️ **Step 4**: We then use the spectrogram to classify using selected model  
➡️ **Step 5**: You get a 🎧 **Predicted Music Genre** with confidence
""")

# ====== MODEL TYPE SELECTION ======
model_type = st.selectbox("Select Model Type", [
    "CNN (cnn_baseline.keras)",
    "CNN (finetuned_model.keras)",
    "Logistic Regression (.pkl)",
    "Random Forest (.pkl)",
    "SVM (.pkl)"
])

# ====== MODEL PERFORMANCE METRICS ======
model_performance = {
    "CNN (cnn_baseline.keras)": "Validation Accuracy: 70.26%, Test Accuracy: 70.48%",
    "CNN (finetuned_model.keras)": "Validation Accuracy: 75.15%, Test Accuracy: 70.5%",
    "Logistic Regression (.pkl)": "Validation Accuracy: 86.99%, Test Accuracy: 88.19%",
    "Random Forest (.pkl)": "Validation Accuracy: 84.39%, Test Accuracy: 86.72%",
    "SVM (.pkl)": "Validation Accuracy: 78.44%, Test Accuracy: 80.07%"
}

# ====== MODEL LOADING ======
cnn_model = None
ml_model = None
idx_to_class = None

if model_type == "CNN (cnn_baseline.keras)":
    MODEL_PATH = "models/cnn_baseline.keras"
    CLASS_INDEX_PATH = "models/class_indices.pkl"
    cnn_model = load_model(MODEL_PATH)
    with open(CLASS_INDEX_PATH, "rb") as f:
        class_indices = joblib.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

elif model_type == "CNN (finetuned_model.keras)":
    MODEL_PATH = "models/finetuned_model.keras"
    CLASS_INDEX_PATH = "models/class_indices.pkl"
    cnn_model = load_model(MODEL_PATH)
    with open(CLASS_INDEX_PATH, "rb") as f:
        class_indices = joblib.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

elif model_type == "Logistic Regression (.pkl)":
    MODEL_PATH = "models/logistic_regression.pkl"
    if os.path.exists(MODEL_PATH):
        ml_model = joblib.load(MODEL_PATH)
    else:
        st.error(f"Model not found: {MODEL_PATH}")

elif model_type == "Random Forest (.pkl)":
    MODEL_PATH = "models/random_forest.pkl"
    if os.path.exists(MODEL_PATH):
        ml_model = joblib.load(MODEL_PATH)
    else:
        st.error(f"Model not found: {MODEL_PATH}")

elif model_type == "SVM (.pkl)":
    MODEL_PATH = "models/svm.pkl"
    if os.path.exists(MODEL_PATH):
        ml_model = joblib.load(MODEL_PATH)
    else:
        st.error(f"Model not found: {MODEL_PATH}")

# ====== SHOW PERFORMANCE IF SELECTED MODEL HAS IT ======
if model_type in model_performance:
    st.info(f"📈 **{model_type} Performance** — {model_performance[model_type]}")

# ====== SIDEBAR INFO ======
with st.sidebar:
    st.header("ℹ️ Info")
    st.markdown("This app converts your audio into a **Mel spectrogram** and classifies it using your selected model.")
    if idx_to_class:
        st.markdown("**Supported Genres:**")
        for genre in sorted(idx_to_class.values()):
            st.markdown(f"- {genre.capitalize()}")

# ====== FILE UPLOAD ======
uploaded_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

if uploaded_audio:
    with st.spinner("Processing audio and generating prediction..."):
        with tempfile.NamedTemporaryFile(suffix=uploaded_audio.name[-4:], delete=False) as tmp_audio:
            tmp_audio.write(uploaded_audio.read())
            audio_path = tmp_audio.name

        if uploaded_audio.name.endswith(".mp3"):
            audio = AudioSegment.from_mp3(audio_path)
            wav_path = audio_path.replace(".mp3", ".wav")
            audio.export(wav_path, format="wav")
        else:
            wav_path = audio_path

        y, sr = librosa.load(wav_path, duration=30)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        librosa.display.specshow(S_dB, sr=sr, cmap='magma', ax=ax)
        ax.axis('off')
        plt.tight_layout(pad=0)

        if model_type.startswith("CNN"):
            tmp_img_path = wav_path.replace(".wav", "_cnn.png")
            plt.savefig(tmp_img_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            img = image.load_img(tmp_img_path, target_size=(300, 400))
            x = image.img_to_array(img) / 255.0
            x = np.expand_dims(x, axis=0)

            preds = cnn_model.predict(x)
            predicted_idx = np.argmax(preds)
            predicted_class = idx_to_class[predicted_idx] if idx_to_class else predicted_idx
            confidence = np.max(preds)

            with st.expander("📊 Show detailed prediction chart"):
                top_n = 3
                top_indices = preds[0].argsort()[-top_n:][::-1]
                top_classes = [idx_to_class[i] for i in top_indices]
                top_scores = [preds[0][i] for i in top_indices]

                df = pd.DataFrame({
                    'Genre': top_classes,
                    'Confidence': [round(score * 100, 2) for score in top_scores]
                })

                fig = px.bar(
                    df, x='Confidence', y='Genre',
                    orientation='h', text='Confidence',
                    color='Genre', color_discrete_sequence=px.colors.sequential.Magma_r
                )
                fig.update_layout(
                    showlegend=False,
                    xaxis=dict(title="Confidence (%)"),
                    title="Top 3 Genre Predictions"
                )

                st.plotly_chart(fig, use_container_width=True)

            st.success(f"🎧 **Predicted Genre:** {predicted_class.title()}")
            st.write(f"🔍 **Confidence:** `{confidence:.2%}`")
            st.audio(uploaded_audio)

        else:
            tmp_img_path = wav_path.replace(".wav", "_ml.png")
            plt.savefig(tmp_img_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            img = Image.open(tmp_img_path).convert('L')
            img = img.resize((128, 128))
            img_array = np.asarray(img).flatten().reshape(1, -1)

            predicted_idx = ml_model.predict(img_array)[0]
            genre_map = {
                0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
                5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'
            }
            genre_label = genre_map.get(predicted_idx, str(predicted_idx))
            if hasattr(ml_model, 'predict_proba'):
                proba = ml_model.predict_proba(img_array)[0]
                confidence = np.max(proba)
                st.success(f"🎷 Predicted Genre: {genre_label.title()}")
                st.write(f"🔍 Confidence: `{confidence:.2%}`")
                st.audio(uploaded_audio)
            else:
                st.success(f"🎷 Predicted Genre: {genre_label.title()}")
                st.warning("⚠️ Confidence not available for this model.")
                st.audio(uploaded_audio)


        os.remove(audio_path)
        if wav_path != audio_path and os.path.exists(wav_path):
            os.remove(wav_path)
        if os.path.exists(tmp_img_path):
            os.remove(tmp_img_path)

# ====== FOOTER ======
st.markdown("---")
st.caption("🎶 Built with Streamlit, TensorFlow, and Librosa")
st.caption("📊 Model trained on the GTZAN dataset")
