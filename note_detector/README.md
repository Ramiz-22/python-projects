# Piano Note Detector

A Python application for real-time detection of piano notes using your microphone with signal processing and Fast Fourier Transform (FFT).

## Features

- **Real-time Piano Note Detection**: Identifies piano notes from A0 to C8
- **Graphical User Interface**: Displays detected notes in a Tkinter window
- **Noise Filtering**: Removes background noise for more accurate results
- **Stabilization Buffer**: Uses a buffer to stabilize output detection

## Prerequisites

- Python 3.8+
- System microphone

## Installation

### 1. Create a Virtual Environment

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
.\venv\Scripts\activate.bat
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install numpy scipy pyaudio
```

## Usage

### Running the Application

```bash
python piano_note_detector.py
```

Or if the virtual environment is not activated:

**Windows (PowerShell):**

```powershell
.\venv\Scripts\python.exe piano_note_detector.py
```

**Windows (Command Prompt):**

```cmd
.\venv\Scripts\python.exe piano_note_detector.py
```

**Linux/macOS:**

```bash
./venv/bin/python piano_note_detector.py
```

### How It Works

1. Run the application
2. A GUI window opens
3. Detected notes are displayed in the window
4. Close the window to stop the program

## Configurable Parameters

All important parameters are located in the `Constants` section of `piano_note_detector.py`:

```python
# Audio buffer size (larger = better frequency resolution but more latency)
CHUNK = 8192

# Sampling rate (Hz)
RATE = 44100

# Noise threshold
NOISE_THRESHOLD = 0.15

# Size of the note stabilization buffer
NOTE_BUFFER_SIZE = 8
```

## Program Architecture

### `audio_callback()` Function

- Receives audio data from the microphone
- Normalizes and applies Hanning window
- Computes FFT with high frequency resolution
- Identifies peak frequency
- Detects corresponding note

### `get_note_from_frequency()` Function

- Converts frequency (Hz) to piano note name
- Implements priority logic for octaves 2 and 3

### `PianoNoteDetector` Class

- Manages GUI interface
- Manages audio stream
- Updates display in real-time

## Processing Flow Diagram

```
Microphone
    ↓
Normalization & Windowing
    ↓
FFT (Fourier Transform)
    ↓
Boost octaves 2 & 3
    ↓
Apply Smoothing
    ↓
Peak Detection
    ↓
Note Detection
    ↓
Stabilization Buffer
    ↓
GUI Display
```

## Troubleshooting

### No Sound is Being Detected

- Check if your microphone is working
- Increase system volume
- Decrease the `NOISE_THRESHOLD` value

### Invalid Notes are Being Detected

- Increase `NOTE_BUFFER_SIZE`
- Increase `NOISE_THRESHOLD`
- Change microphone placement

### High Latency in Detection

- Decrease `CHUNK` size (lower resolution but less latency)
- Decrease `NOTE_BUFFER_SIZE`

## Dependencies

| Package | Version | Purpose                                     |
| ------- | ------- | ------------------------------------------- |
| numpy   | 2.3.5+  | Numerical computations and array processing |
| scipy   | 1.16.3+ | FFT and signal processing operations        |
| pyaudio | 0.2.14+ | Access to audio devices                     |

## License

This project is for educational purposes.

## Author

Piano Note Detector with Signal Processing and GUI

## Helpful Tips

- The program works best in quiet environments
- Octaves 2 and 3 are boosted for better accuracy
- Using a high-quality microphone is recommended
- The program runs audio processing in a separate thread to keep the GUI responsive
