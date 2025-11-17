import numpy as np

def sine_wave(frequency, duration, sample_rate=44100, amplitude=1.0, phase=0.0):
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)
