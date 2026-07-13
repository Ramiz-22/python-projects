# Python Projects

A monorepo containing two independent Python GUI applications.

## Projects

### 1. Piano Note Detector (`note_detector/`)

A real-time piano note detection application that listens to microphone input and displays the detected musical note on a GUI.

**How it works:**
- Captures audio via microphone at 44.1 kHz using PyAudio
- Applies FFT with Hanning windowing and spectral smoothing
- Detects the fundamental frequency and maps it to the nearest piano note (A0 to C8)
- Features octave-specific accuracy improvements and dynamic noise thresholding
- Displays the stable note in a Tkinter window with a rolling buffer for flicker reduction

**Dependencies:** `numpy`, `scipy`, `pyaudio`

```bash
cd note_detector
pip install numpy scipy pyaudio
python piano_note_detector.py
```

### 2. Screen Brightness Control (`screen-brightness-control/`)

A cross-platform desktop GUI application for viewing and adjusting monitor brightness.

**Features:**
- Displays current brightness level of all connected monitors
- Set brightness via text entry (0-100) or Enter key
- Up/Down arrow keys adjust brightness in 5-step increments with debounce
- Multi-monitor support
- System theme awareness (light/dark mode)

**Dependencies:** `customtkinter`, `screen-brightness-control`

```bash
cd screen-brightness-control
pip install customtkinter screen-brightness-control
python brightness_controller.pyw
```
