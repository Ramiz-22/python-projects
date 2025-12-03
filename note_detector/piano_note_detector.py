import numpy as np
import pyaudio
import tkinter as tk
from scipy.fftpack import fft
import queue
from typing import Tuple

# --- Constants for Audio Processing ---

# Number of audio frames per buffer.
# A larger chunk size increases latency but can improve frequency resolution.
CHUNK = 8192

# Audio format (32-bit floating point).
FORMAT = pyaudio.paFloat32

# Number of audio channels (1 for mono).
CHANNELS = 1

# Sampling rate in Hz (samples per second).
RATE = 44100

# Size of the analysis window for FFT.
WINDOW_SIZE = 8192

# Threshold to filter out background noise.
# Amplitudes below this are considered noise.
NOISE_THRESHOLD = 0.15

# --- Constants for Note Detection ---

# Minimum and maximum frequencies to detect, corresponding to the piano range (A0 to C8).
MIN_FREQUENCY = 27.5
MAX_FREQUENCY = 4186.0

# Number of recent notes to store in a buffer for stabilizing the output.
# The most common note in the buffer is displayed.
NOTE_BUFFER_SIZE = 8

# --- Note Frequency Mapping ---

# A dictionary mapping standard piano note names to their fundamental frequencies in Hz.
# This covers the full 88-key piano range from A0 to C8.
PIANO_NOTES = {
    # Octave 0
    'A0': 27.50, 'A#0': 29.14, 'B0': 30.87,
    # Octave 1
    'C1': 32.70, 'C#1': 34.65, 'D1': 36.71, 'D#1': 38.89, 'E1': 41.20, 'F1': 43.65, 'F#1': 46.25, 'G1': 49.00, 'G#1': 51.91, 'A1': 55.00, 'A#1': 58.27, 'B1': 61.74,
    # Octave 2
    'C2': 65.41, 'C#2': 69.30, 'D2': 73.42, 'D#2': 77.78, 'E2': 82.41, 'F2': 87.31, 'F#2': 92.50, 'G2': 98.00, 'G#2': 103.83, 'A2': 110.00, 'A#2': 116.54, 'B2': 123.47,
    # Octave 3
    'C3': 130.81, 'C#3': 138.59, 'D3': 146.83, 'D#3': 155.56, 'E3': 164.81, 'F3': 174.61, 'F#3': 185.00, 'G3': 196.00, 'G#3': 207.65, 'A3': 220.00, 'A#3': 233.08, 'B3': 246.94,
    # Octave 4
    'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13, 'E4': 329.63, 'F4': 349.23, 'F#4': 369.99, 'G4': 392.00, 'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88,
    # Octave 5
    'C5': 523.25, 'C#5': 554.37, 'D5': 587.33, 'D#5': 622.25, 'E5': 659.25, 'F5': 698.46, 'F#5': 739.99, 'G5': 783.99, 'G#5': 830.61, 'A5': 880.00, 'A#5': 932.33, 'B5': 987.77,
    # Octave 6
    'C6': 1046.50, 'C#6': 1108.73, 'D6': 1174.66, 'D#6': 1244.51, 'E6': 1318.51, 'F6': 1396.91, 'F#6': 1479.98, 'G6': 1567.98, 'G#6': 1661.22, 'A6': 1760.00, 'A#6': 1864.66, 'B6': 1975.53,
    # Octave 7
    'C7': 2093.00, 'C#7': 2217.46, 'D7': 2349.32, 'D#7': 2489.02, 'E7': 2637.02, 'F7': 2793.83, 'F#7': 2959.96, 'G7': 3135.96, 'G#7': 3322.44, 'A7': 3520.00, 'A#7': 3729.31, 'B7': 3951.07,
    # Octave 8
    'C8': 4186.01
}

# --- Global Variables ---
# A queue to safely pass detected notes from the audio thread to the main GUI thread.
note_queue = queue.Queue()
# A buffer to store the last few detected notes to stabilize the output.
note_buffer = []


def get_note_from_frequency(frequency: float) -> str:
    """
    Finds the closest musical note corresponding to a given frequency.
    This version includes specific logic to improve accuracy for octaves 2 and 3.

    Args:
        frequency: The input frequency in Hz.

    Returns:
        The name of the closest note (e.g., "A4") or "---" if the frequency is zero.
    """
    if frequency == 0:
        return "---"

    # Define frequency ranges for specific octaves to prioritize them.
    octave_ranges = {
        2: (65.41, 123.47),   # C2 to B2
        3: (130.81, 246.94),  # C3 to B3
    }

    # Determine if the frequency falls within one of the prioritized octaves.
    current_octave = None
    for octave, (min_freq, max_freq) in octave_ranges.items():
        if min_freq <= frequency <= max_freq:
            current_octave = octave
            break

    # If in a prioritized octave, search only within that octave's notes.
    if current_octave:
        octave_notes = {k: v for k, v in PIANO_NOTES.items()
                       if k.endswith(str(current_octave))}
        # Find the note with the minimum frequency difference.
        closest_note = min(octave_notes.items(),
                         key=lambda x: abs(frequency - x[1]))
    else:
        # Otherwise, search through all piano notes.
        closest_note = min(PIANO_NOTES.items(),
                         key=lambda x: abs(frequency - x[1]))

    return closest_note[0]

def audio_callback(in_data: bytes, frame_count: int, time_info: dict, status: int) -> Tuple[bytes, int]:
    """
    This function is called by PyAudio for each new buffer of audio data.
    It performs the core signal processing to detect the fundamental frequency.

    Args:
        in_data: The raw audio data buffer.
        frame_count: The number of frames in the buffer.
        time_info: A dictionary containing timestamp information.
        status: A flag indicating any audio stream errors.

    Returns:
        A tuple containing the original data and a flag to continue the stream.
    """
    try:
        # Convert raw byte data to a NumPy array of floats.
        audio_data = np.frombuffer(in_data, dtype=np.float32)

        if len(audio_data) == 0:
            note_queue.put("---")
            return (in_data, pyaudio.paContinue)

        # 1. Normalization and Windowing
        # Normalize audio to a [-1, 1] range to handle varying input levels.
        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-10)
        # Apply a Hanning window to reduce spectral leakage during FFT.
        window = np.hanning(len(audio_data))
        audio_data = audio_data * window

        # 2. FFT (Fast Fourier Transform)
        # Pad the audio data and use a larger window for better low-frequency resolution.
        padded_window = np.hanning(CHUNK * 2)
        padded_audio = np.pad(audio_data, (0, len(padded_window) - len(audio_data)))
        windowed_audio = padded_audio * padded_window
        
        # Increase FFT points for higher frequency resolution.
        fft_data = fft(windowed_audio, n=CHUNK * 8)
        # Calculate the frequency axis for the FFT result.
        freqs = np.fft.fftfreq(len(fft_data), 1.0/RATE)
        # Get the magnitude spectrum (we only need the positive frequencies).
        magnitude_spectrum = np.abs(fft_data[:len(fft_data)//2])

        # 3. Peak Detection Enhancement
        # Find the initial peak to guide further processing.
        initial_peak_index = np.argmax(magnitude_spectrum)
        peak_frequency = abs(freqs[initial_peak_index])

        # Boost the magnitude of frequencies in the 2nd and 3rd octaves.
        octave_2_3_mask = (65 <= abs(freqs[:len(magnitude_spectrum)])) & \
                         (abs(freqs[:len(magnitude_spectrum)]) <= 250)
        freq_weights = np.ones_like(magnitude_spectrum)
        freq_weights[octave_2_3_mask] *= 2.0
        weighted_spectrum = magnitude_spectrum * freq_weights

        # 4. Smoothing and Final Peak Finding
        # Smooth the spectrum to reduce noise and find a more stable peak.
        # Use a wider smoothing window for lower frequencies.
        smooth_window_size = 51 if peak_frequency < 250 else 31
        smoothed_spectrum = np.convolve(weighted_spectrum,
                                      np.hanning(smooth_window_size),
                                      mode='same')

        # Find the new peak in the smoothed spectrum.
        peak_frequency_index = np.argmax(smoothed_spectrum)
        peak_frequency = abs(freqs[peak_frequency_index])
        peak_magnitude = smoothed_spectrum[peak_frequency_index]

        # 5. Note Identification
        # Dynamically adjust the noise threshold based on frequency.
        adjusted_threshold = NOISE_THRESHOLD * (1.0 + np.exp(-peak_frequency / 200.0))

        # Check if the detected peak is strong enough and within the piano's frequency range.
        if (peak_magnitude > adjusted_threshold and
            MIN_FREQUENCY <= peak_frequency <= MAX_FREQUENCY):

            note = get_note_from_frequency(peak_frequency)

            # Add the detected note to a buffer to stabilize the output.
            global note_buffer
            note_buffer.append(note)
            if len(note_buffer) > NOTE_BUFFER_SIZE:
                note_buffer.pop(0)

            # Determine the most common note in the buffer.
            if note_buffer:
                from collections import Counter
                stable_note = Counter(note_buffer).most_common(1)[0][0]
                note_queue.put(stable_note)
                print(f"Frequency: {peak_frequency:.1f} Hz | Note: {stable_note}")
        else:
            # If no significant note is detected, report as such.
            note_queue.put("---")

    except Exception as e:
        print(f"Error in audio callback: {e}")
        note_queue.put("---")

    return (in_data, pyaudio.paContinue)


class PianoNoteDetector:
    """
    A GUI application that detects and displays piano notes from microphone input.
    """
    def __init__(self):
        """Initializes the Tkinter GUI and the PyAudio stream."""
        self.root = tk.Tk()
        self.root.title("Piano Note Detector")
        self.root.geometry("400x250")

        # Create and configure the label for displaying the detected note.
        self.note_label = tk.Label(
            self.root,
            text="---",
            font=("Helvetica", 60),
            bg="black",
            fg="white"
        )
        self.note_label.pack(expand=True, fill="both")

        # Initialize PyAudio to manage the audio stream.
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=audio_callback  # Process audio in a separate thread.
        )

        # Start the audio stream.
        self.stream.start_stream()

        # Begin the process of updating the GUI.
        self.update_display()

    def update_display(self):
        """
        Periodically checks the note queue for new notes and updates the GUI.
        This method runs on the main GUI thread.
        """
        try:
            # Get a note from the queue without blocking.
            note = note_queue.get_nowait()
            # Update the text of the label.
            self.note_label.config(text=note)
            # Also print the detected note to the console for debugging.
            print(f"Detected Note: {note}")
        except queue.Empty:
            # If the queue is empty, do nothing.
            pass

        # Schedule this method to run again after 50ms.
        self.root.after(50, self.update_display)

    def run(self):
        """Starts the main event loop of the Tkinter application."""
        try:
            self.root.mainloop()
        finally:
            # Cleanly stop and close the audio stream and terminate PyAudio.
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

if __name__ == "__main__":
    # Create an instance of the application and run it.
    app = PianoNoteDetector()
    app.run()