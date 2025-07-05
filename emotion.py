import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
audio_path = "C:/Users/vishn/Data/speech_emotion/angry_01.wav"
if not os.path.exists(audio_path):
    print("❌ File not found. Please check the path.")
    exit()
else:
    print("✅ File loaded:", audio_path)
y, sr = librosa.load(audio_path, duration=3)
print(f"Sample Rate: {sr}, Audio Length: {len(y)} samples")
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_mean = np.mean(mfccs)
chroma = librosa.feature.chroma_stft(y=y, sr=sr)
chroma_mean = np.mean(chroma)
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
mel_db = librosa.power_to_db(mel_spec, ref=np.max)
mel_mean = np.mean(mel_db)
zcr = librosa.feature.zero_crossing_rate(y)
zcr_mean = np.mean(zcr)
centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
centroid_mean = np.mean(centroid)
rmse = librosa.feature.rms(y=y)
rmse_mean = np.mean(rmse)
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
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title("Mel Spectrogram of Speech")
plt.tight_layout()
plt.show()
