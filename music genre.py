import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
audio_path = "C:/Users/vishn/Downloads/GTZAN/genres/blues/blues.00000.wav"
try:
    print(f"\nLoading audio file: {audio_path}")
    y, sr = librosa.load(audio_path, duration=30)
    print(f"Audio loaded successfully! Sample Rate: {sr}, Audio shape: {y.shape}")
except FileNotFoundError:
    print("⚠️ Error: Audio file not found. Please check the file path.")
    exit()
stft = np.abs(librosa.stft(y))
stft_mean = np.mean(stft)
print(f"\nSTFT Mean Amplitude: {stft_mean:.2f}")
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
mel_mean = np.mean(mel_spec_db)
print(f"Mel Spectrogram Mean: {mel_mean:.2f} dB")
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
mfcc_mean = np.mean(mfccs)
print(f"MFCC Mean: {mfcc_mean:.2f}")
features = {
    "STFT_Mean": stft_mean,
    "MelSpectrogram_Mean": mel_mean,
    "MFCC_Mean": mfcc_mean
}
print("\n📊 Structured Audio Feature Vector:")
for key, val in features.items():
    print(f"{key}: {val:.2f}")
plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram - Sample Audio')
plt.tight_layout()
plt.show()
