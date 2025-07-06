# 🎤 Multilingual Transcriber

A real-time speech-to-text transcription application built with OpenAI's Whisper model and Python. This application provides live transcription with support for multiple languages and a user-friendly GUI interface.

## ✨ Features

- **Real-time Transcription**: Live audio recording and transcription
- **Multilingual Support**: Automatic language detection and transcription
- **Modern GUI**: Clean, intuitive interface built with tkinter
- **GPU Acceleration**: CUDA support for faster transcription (when available)
- **Timestamp Display**: Each transcription includes timestamps
- **Audio Chunking**: Efficient processing of audio in short segments
- **Cross-platform**: Works on Windows, macOS, and Linux

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Microphone access
- (Optional) NVIDIA GPU with CUDA support for faster processing

### Installation

#### Option 1: Using Conda (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/BadassAman4014/Real-time-Bilingual-GenAI-Speech-Transcript-Generator.git
   cd "Multilingual Trancriber"
   ```

2. **Create and activate the conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate whisper
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

#### Option 2: Using pip

1. **Clone the repository**
   ```bash
   git clone # 🎤 Multilingual Transcriber

A real-time speech-to-text transcription application built with OpenAI's Whisper model and Python. This application provides live transcription with support for multiple languages and a user-friendly GUI interface.

## ✨ Features

- **Real-time Transcription**: Live audio recording and transcription
- **Multilingual Support**: Automatic language detection and transcription
- **Modern GUI**: Clean, intuitive interface built with tkinter
- **GPU Acceleration**: CUDA support for faster transcription (when available)
- **Timestamp Display**: Each transcription includes timestamps
- **Audio Chunking**: Efficient processing of audio in short segments
- **Cross-platform**: Works on Windows, macOS, and Linux

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Microphone access
- (Optional) NVIDIA GPU with CUDA support for faster processing

### Installation

#### Option 1: Using Conda (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/BadassAman4014/Real-time-Bilingual-GenAI-Speech-Transcript-Generator.git
   cd "Multilingual Trancriber"
   ```

2. **Create and activate the conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate whisper
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

#### Option 2: Using pip

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Multilingual Trancriber"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

## 📖 Usage

1. **Launch the application** by running `python app.py`
2. **Click "🎤 Start"** to begin recording and transcription
3. **Speak clearly** into your microphone
4. **View transcriptions** in real-time in the text area
5. **Click "⏹ Stop"** to stop recording
6. **Use "🗑 Clear"** to clear the transcription history

### Status Indicators

- ⚪ **Idle**: Application is ready but not recording
- 🔴 **Recording**: Currently recording audio
- ⏳ **Transcribing**: Processing audio with Whisper
- 🎧 **Listening**: Ready for the next audio chunk
- ⚠️ **Error**: An error occurred during processing

## 🏗️ Project Structure

```
Multilingual Trancriber/
├── app.py                 # Main entry point
├── requirements.txt       # Python dependencies
├── environment.yml        # Conda environment
├── README.md             # This file
└── transcriber/          # Core application package
    ├── __init__.py       # Package initialization
    ├── audio.py          # Audio recording functionality
    ├── model.py          # Whisper model management
    └── gui.py            # GUI interface
```

### Code Organization

- **`app.py`**: Entry point that launches the GUI
- **`transcriber/audio.py`**: Handles audio recording and processing
- **`transcriber/model.py`**: Manages Whisper model loading and inference
- **`transcriber/gui.py`**: Contains the main GUI application class

## 🔧 Configuration

### Audio Settings

The application uses the following audio configuration:
- **Sample Rate**: 16,000 Hz
- **Channels**: 1 (Mono)
- **Format**: 16-bit PCM
- **Chunk Size**: 1024 samples
- **Recording Duration**: 4 seconds per chunk

### Model Settings

- **Model**: OpenAI Whisper "base" model
- **Device**: Automatically detects CUDA GPU or falls back to CPU
- **Language**: Auto-detection enabled

## 🛠️ Development

### Adding New Features

1. **Audio Processing**: Modify `transcriber/audio.py`
2. **Model Configuration**: Update `transcriber/model.py`
3. **GUI Changes**: Edit `transcriber/gui.py`

### Testing

To test the application:
1. Ensure your microphone is working
2. Run the application
3. Speak in different languages to test multilingual support
4. Check that transcriptions appear with timestamps

## 📋 Dependencies

### Core Dependencies
- `whisper`: OpenAI's speech recognition model
- `torch`: PyTorch for deep learning
- `pyaudio`: Audio recording and playback
- `tkinter`: GUI framework (included with Python)

### Additional Dependencies
- `numpy`: Numerical computing
- `librosa`: Audio processing
- `soundfile`: Audio file handling
- And many others (see `requirements.txt` for complete list)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech recognition model
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) for audio processing

## 🐛 Troubleshooting

### Common Issues

1. **"Could not open audio stream"**
   - Check microphone permissions
   - Ensure microphone is not in use by another application

2. **Slow transcription**
   - Install CUDA for GPU acceleration
   - Use a smaller Whisper model (e.g., "tiny" instead of "base")

3. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Activate the correct conda environment: `conda activate whisper`

### Performance Tips

- Use a GPU for faster transcription
- Close other applications using the microphone
- Speak clearly and minimize background noise
- Consider using a smaller Whisper model for faster processing

## 📞 Support

If you encounter any issues or have questions, please:
1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed information about your problem

---

**Happy Transcribing! 🎤✨** 
   cd "Multilingual Trancriber"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

## 📖 Usage

1. **Launch the application** by running `python app.py`
2. **Click "🎤 Start"** to begin recording and transcription
3. **Speak clearly** into your microphone
4. **View transcriptions** in real-time in the text area
5. **Click "⏹ Stop"** to stop recording
6. **Use "🗑 Clear"** to clear the transcription history

### Status Indicators

- ⚪ **Idle**: Application is ready but not recording
- 🔴 **Recording**: Currently recording audio
- ⏳ **Transcribing**: Processing audio with Whisper
- 🎧 **Listening**: Ready for the next audio chunk
- ⚠️ **Error**: An error occurred during processing

## 🏗️ Project Structure

```
Multilingual Trancriber/
├── app.py                 # Main entry point
├── requirements.txt       # Python dependencies
├── environment.yml        # Conda environment
├── README.md             # This file
└── transcriber/          # Core application package
    ├── __init__.py       # Package initialization
    ├── audio.py          # Audio recording functionality
    ├── model.py          # Whisper model management
    └── gui.py            # GUI interface
```

### Code Organization

- **`app.py`**: Entry point that launches the GUI
- **`transcriber/audio.py`**: Handles audio recording and processing
- **`transcriber/model.py`**: Manages Whisper model loading and inference
- **`transcriber/gui.py`**: Contains the main GUI application class

## 🔧 Configuration

### Audio Settings

The application uses the following audio configuration:
- **Sample Rate**: 16,000 Hz
- **Channels**: 1 (Mono)
- **Format**: 16-bit PCM
- **Chunk Size**: 1024 samples
- **Recording Duration**: 4 seconds per chunk

### Model Settings

- **Model**: OpenAI Whisper "base" model
- **Device**: Automatically detects CUDA GPU or falls back to CPU
- **Language**: Auto-detection enabled

## 🛠️ Development

### Adding New Features

1. **Audio Processing**: Modify `transcriber/audio.py`
2. **Model Configuration**: Update `transcriber/model.py`
3. **GUI Changes**: Edit `transcriber/gui.py`

### Testing

To test the application:
1. Ensure your microphone is working
2. Run the application
3. Speak in different languages to test multilingual support
4. Check that transcriptions appear with timestamps

## 📋 Dependencies

### Core Dependencies
- `whisper`: OpenAI's speech recognition model
- `torch`: PyTorch for deep learning
- `pyaudio`: Audio recording and playback
- `tkinter`: GUI framework (included with Python)

### Additional Dependencies
- `numpy`: Numerical computing
- `librosa`: Audio processing
- `soundfile`: Audio file handling
- And many others (see `requirements.txt` for complete list)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for the speech recognition model
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) for audio processing

## 🐛 Troubleshooting

### Common Issues

1. **"Could not open audio stream"**
   - Check microphone permissions
   - Ensure microphone is not in use by another application

2. **Slow transcription**
   - Install CUDA for GPU acceleration
   - Use a smaller Whisper model (e.g., "tiny" instead of "base")

3. **Import errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Activate the correct conda environment: `conda activate whisper`

### Performance Tips

- Use a GPU for faster transcription
- Close other applications using the microphone
- Speak clearly and minimize background noise
- Consider using a smaller Whisper model for faster processing

## 📞 Support

If you encounter any issues or have questions, please:
1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed information about your problem

---

**Happy Transcribing! 🎤✨** 
