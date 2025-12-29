import random
import torchaudio.transforms as T

def augment(waveform):
    if random.random() < 0.5:
        waveform = T.TimeStretch()(waveform)
    if random.random() < 0.5:
        waveform += 0.005 * torch.randn_like(waveform)
    return waveform
