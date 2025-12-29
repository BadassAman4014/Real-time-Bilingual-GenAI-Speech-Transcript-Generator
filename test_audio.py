import librosa
audio, sr = librosa.load("data/raw/en/test_en_16k.wav", sr=16000)
print(len(audio), sr)