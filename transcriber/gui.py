import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
from datetime import datetime
import threading
import os
import queue
from .audio import AudioRecorder
from .model import load_model
import torch

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎤 Whisper Real-Time Transcriber")
        self.root.geometry("820x600")
        self.root.resizable(False, False)
        self.recorder = None
        self.is_recording = False
        self.transcriptions = []
        self.model, self.device = load_model()
        self.setup_ui()
        self.root.after(500, self.update_gui_loop)
        threading.Thread(target=self.transcription_worker, daemon=True).start()

    def setup_ui(self):
        style = ttk.Style(self.root)
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        self.textbox = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, font=("Consolas", 11), width=90, height=25, bg="#f9f9f9")
        self.textbox.pack(padx=12, pady=10, fill=tk.BOTH, expand=True)
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)
        self.start_btn = ttk.Button(btn_frame, text="🎤 Start", command=self.start_recording)
        self.start_btn.grid(row=0, column=0, padx=6)
        self.stop_btn = ttk.Button(btn_frame, text="⏹ Stop", command=self.stop_recording, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=6)
        self.clear_btn = ttk.Button(btn_frame, text="🗑 Clear", command=self.clear_text)
        self.clear_btn.grid(row=0, column=2, padx=6)
        self.status_var = tk.StringVar()
        self.status_var.set(f"⚪ Idle | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W, font=("Segoe UI", 9), bg="#eee")
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def start_recording(self):
        self.recorder = AudioRecorder()
        self.recorder.start_recording()
        self.is_recording = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("🔴 Recording... Listening for speech.")

    def stop_recording(self):
        if self.recorder:
            self.recorder.stop_recording()
        self.is_recording = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("⚪ Stopped recording")

    def clear_text(self):
        self.textbox.delete(1.0, tk.END)
        self.transcriptions = []

    def update_gui_loop(self):
        self.root.after(500, self.update_gui_loop)

    def transcription_worker(self):
        while True:
            if self.recorder:
                try:
                    audio_path = self.recorder.audio_queue.get(timeout=1)
                    self.status_var.set("⏳ Transcribing...")
                    result = self.model.transcribe(audio_path)
                    transcription = str(result.get('text', '')).strip()
                    language = str(result.get('language', 'en'))
                    if transcription:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        entry = f"[{timestamp}] ({language.upper()}) {transcription}\n"
                        self.transcriptions.append(entry)
                        self.textbox.insert(tk.END, entry)
                        self.textbox.see(tk.END)
                    self.status_var.set("🎧 Listening...")
                    os.remove(audio_path)
                except queue.Empty:
                    continue
                except Exception as e:
                    self.status_var.set("⚠️ Error")
                    messagebox.showerror("Transcription Error", str(e)) 