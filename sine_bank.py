import numpy as np
from sine import sine_wave   # <-- this pulls in your sine_wave function

def sine_bank(frequencies, duration, sample_rate=44100, amplitudes=None):
    """
    Build a sine-bank oscillator.
    
    frequencies: list of frequencies in Hz
    amplitudes: list of amplitudes (same length as frequencies)
                if None → all sines use amplitude = 1.0
    """
    if amplitudes is None:
        amplitudes = [1.0] * len(frequencies)

    # start with silence
    output = np.zeros(int(duration * sample_rate), dtype=float)

    for f, a in zip(frequencies, amplitudes):
        output += sine_wave(f, duration, sample_rate, amplitude=a)

    return output
  
# ----------------------------------------------------------
# Example use
# ----------------------------------------------------------

if __name__ == "__main__":
    sr = 44100
    duration = 2.0

    # A harmonic series for A2 (110 Hz)
    freqs = [110 * n for n in range(1, 9)]   # 8 harmonics
    amps = [1/n for n in range(1, 9)]        # simple rolloff

    y = sine_bank(freqs, duration, sample_rate=sr, amplitudes=amps)

    print("Generated sine bank with", len(freqs), "oscillators.")
    print("Output samples:", len(y))
