# 🎵 Music Genre Classification Web App

This repository hosts a **Streamlit-based web application** for automatic music genre classification. Users can upload audio files (`.mp3` or `.wav`), which are then converted to spectrograms and classified into genres using either a **Convolutional Neural Network (CNN)** or traditional **machine learning (ML)** models.

---

## 🔍 About the Project

This application is part of a broader project on audio signal processing and deep learning. It demonstrates how Mel spectrograms — visual representations of sound — can be used as inputs to both image-based CNN models and classic ML classifiers.

The web app is designed to be lightweight, interactive, and deployable in cloud environments like Streamlit Cloud.

---

## ✨ Features

- **Model Selection**: Choose between CNN (`.keras`) and ML (`.pkl`) models (Logistic Regression, Random Forest, SVM).
- **Audio Upload**: Supports `.mp3` and `.wav` formats with drag-and-drop or file browser.
- **Spectrogram Generation**: Uses Librosa to convert uploaded audio into Mel spectrograms.
- **Dual Pipeline Handling**:
  - CNNs receive RGB spectrogram images (300×400)
  - ML models receive flattened grayscale images (128×128)
- **Genre Prediction**:
  - Displays the predicted genre label and model confidence
  - For CNNs, shows Top-3 predictions using an interactive Plotly chart
- **Audio Playback**: Users can listen to their uploaded file in the browser
- **Waveform and Spectrogram Visualization**
- **Model Performance Display**: Validation and test accuracy shown for selected model

---

## 🧠 Technologies Used

- Python 3.12
- TensorFlow / Keras (CNN)
- Scikit-learn (ML models)
- Librosa (audio processing)
- Streamlit (web UI)
- Plotly (visualization)
- Pydub (MP3 decoding)

---

## 🚀 Deployment

This app is deployed on Streamlit Cloud:
👉 [Launch the App](https://myahninsi-music-genre-classification-cnn-streamlit-app-dlavyq.streamlit.app/)

To run it locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Ensure `ffmpeg` is installed locally if you want to support `.mp3` uploads:
- Windows: `choco install ffmpeg`
- macOS: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

---

## 📁 Repository Structure
```
├── models/                   # Trained models (.keras, .pkl, class_indices.pkl)
├── app.py                   # Main Streamlit app
├── README.md                # Project overview and documentation
├── LICENSE                  # Open source license
├── requirements.txt         # Python dependencies
├── packages.txt             # System dependencies for Streamlit Cloud
```

---

## 🎧 Example Genre Labels
- Rock
- Pop
- Jazz
- Classical
- Hip-hop
- Disco
- Reggae
- Blues
- Metal
- Country

---

## 📚 Dataset
This app was trained on the **GTZAN Genre Collection** — a benchmark dataset consisting of 10 genres with 100 audio tracks each.

---

## 🛠️ Credits
Built by myahninsi as part of a course project on deep learning and audio classification. Special thanks to Librosa and Streamlit for making audio ML easier and more visual.