import pyaudio
import wave
import tempfile
import queue
from tkinter import messagebox

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 4  # Short chunk for real-time

class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.recording = False
        self.audio_queue = queue.Queue()

    def start_recording(self):
        self.recording = True
        self.frames = []

        def record():
            try:
                stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            except Exception as e:
                messagebox.showerror("Audio Error", f"Could not open audio stream:\n{e}")
                return

            while self.recording:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    self.frames.append(data)
                    if len(self.frames) >= RATE * RECORD_SECONDS // CHUNK:
                        self.save_audio_chunk()
                        self.frames = []
                except Exception as e:
                    messagebox.showerror("Recording Error", str(e))
                    break

            stream.stop_stream()
            stream.close()

        import threading
        threading.Thread(target=record, daemon=True).start()

    def stop_recording(self):
        self.recording = False
        self.audio.terminate()

    def save_audio_chunk(self):
        if not self.frames:
            return
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wf = wave.open(tmp.name, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            self.audio_queue.put(tmp.name) 