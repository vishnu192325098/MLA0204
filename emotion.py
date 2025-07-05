import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# === Step 1: Provide path to the speech file ===
audio_path = "C:/Users/vishn/Data/speech_emotion/angry_01.wav"  # Replace with your file

# === Step 2: Check if file exists ===
if not os.path.exists(audio_path):
    print("❌ File not found. Please check the path.")
    exit()
else:
    print("✅ File loaded:", audio_path)

# === Step 3: Load the audio file ===
y, sr = librosa.load(audio_path, duration=3)
print(f"Sample Rate: {sr}, Audio Length: {len(y)} samples")

# === Step 4: Extract Acoustic Features ===

# MFCCs
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_mean = np.mean(mfccs)

# Chroma
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
chroma_mean = np.mean(chroma)

# Mel Spectrogram
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
mel_db = librosa.power_to_db(mel_spec, ref=np.max)
mel_mean = np.mean(mel_db)

# Zero-Crossing Rate
zcr = librosa.feature.zero_crossing_rate(y)
zcr_mean = np.mean(zcr)

# Spectral Centroid
centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
centroid_mean = np.mean(centroid)

# RMS Energy
rmse = librosa.feature.rms(y=y)
rmse_mean = np.mean(rmse)

# === Step 5: Print Features as Structured Output ===
features = {
    "MFCC_Mean": mfcc_mean,
    "Chroma_Mean": chroma_mean,
    "MelSpectrogram_Mean": mel_mean,
    "ZCR_Mean": zcr_mean,
    "Spectral_Centroid_Mean": centroid_mean,
    "RMS_Energy_Mean": rmse_mean
}

print("\n📊 Extracted Acoustic Features:")
for key, val in features.items():
    print(f"{key}: {val:.4f}")

# === Step 6: Plot Mel Spectrogram ===
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram of Speech")
plt.tight_layout()
plt.show()
