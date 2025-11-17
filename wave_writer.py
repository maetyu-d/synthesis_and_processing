import numpy as np
from waveforms import sine_wave   # import the sine oscillator
import wave, struct               # needed for the WAV writer


# ----------------------------
# 8-line WAV writer (inline)
# ----------------------------
def write_wav(filename, signal, sr=44100):
    signal = np.clip(signal, -1, 1)
    data = (signal * 32767).astype(np.int16)
    with wave.open(filename, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.tobytes())


# ----------------------------
# Sine bank (purely for use in our example below)
# ----------------------------
def sine_bank(frequencies, duration, sample_rate=44100, amplitudes=None):
    if amplitudes is None:
        amplitudes = [1.0] * len(frequencies)

    output = np.zeros(int(duration * sample_rate), dtype=float)

    for f, a in zip(frequencies, amplitudes):
        output += sine_wave(f, duration, sample_rate, amplitude=a)

    return output


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    sr = 44100
    duration = 2.0

    # 110 Hz harmonic series
    freqs = [110 * n for n in range(1, 9)]
    amps = [1/n for n in range(1, 9)]

    y = sine_bank(freqs, duration, sr, amps)

    # save it
    write_wav("sine_bank.wav", y, sr)

    print("Wrote sine_bank.wav")
