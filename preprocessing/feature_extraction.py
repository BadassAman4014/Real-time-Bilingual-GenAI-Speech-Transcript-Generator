import torchaudio.transforms as T

mfcc = T.MFCC(sample_rate=16000, n_mfcc=40)
spec = T.MelSpectrogram(sample_rate=16000)

mfcc_features = mfcc(waveform)
spectrogram = spec(waveform)
