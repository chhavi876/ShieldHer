# Voice Stress Detection Model – ShieldHer V2

This AI model is trained to detect stress in voice using speech features from the RAVDESS dataset.

## ✅ Dataset Used
- RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- We used emotional speech clips (not songs)

## ✅ Features Extracted
- 40 MFCCs per audio clip
- Extracted using `librosa` in Python
- Averaged over time for fixed-length input

## ✅ Emotion Categories
- **Stressed**: angry, fearful, sad → labeled `1`
- **Not Stressed**: calm, happy, neutral → labeled `0`

## ✅ Model Architecture
- Dense Neural Network using TensorFlow/Keras
- Layers: 256 → Dropout → 128 → Dropout → 1 (Sigmoid)

## ✅ Output
- Binary Classification:
  - 0 = Not Stressed
  - 1 = Stressed

## ✅ Deployment
- Saved as `.h5` and converted to `.tflite` for mobile integration in the ShieldHer app.
