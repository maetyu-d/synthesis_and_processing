import numpy as np

# -------------------------------------------------------
# Time helper
# -------------------------------------------------------

def _t(duration, sample_rate):
    return np.arange(int(duration * sample_rate)) / sample_rate

def _phase(t, freq, phase):
    return (freq * t + phase / (2 * np.pi)) % 1.0


# -------------------------------------------------------
# Waveforms
# -------------------------------------------------------

def sine_wave(freq, duration, sample_rate=44100, amplitude=1.0, phase=0.0):
    t = _t(duration, sample_rate)
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


def square_wave(freq, duration, sample_rate=44100, amplitude=1.0, phase=0.0):
    t = _t(duration, sample_rate)
    frac = _phase(t, freq, phase)
    return amplitude * np.where(frac < 0.5, 1.0, -1.0)


def pulse_wave(freq, duration, duty=0.2, sample_rate=44100, amplitude=1.0, phase=0.0):
    """
    duty = fraction of period high (0..1)
    """
    t = _t(duration, sample_rate)
    frac = _phase(t, freq, phase)
    return amplitude * np.where(frac < duty, 1.0, -1.0)


def sawtooth_wave(freq, duration, sample_rate=44100, amplitude=1.0, phase=0.0, rising=True):
    t = _t(duration, sample_rate)
    frac = _phase(t, freq, phase)
    wave = 2.0 * frac - 1.0           # rising sawtooth
    if not rising:
        wave = -wave                  # falling sawtooth
    return amplitude * wave


def triangle_wave(freq, duration, sample_rate=44100, amplitude=1.0, phase=0.0):
    t = _t(duration, sample_rate)
    frac = _phase(t, freq, phase)
    tri = 4.0 * np.abs(frac - 0.5) - 1.0
    return amplitude * tri


def noise_wave(duration, sample_rate=44100, amplitude=1.0):
    n = int(duration * sample_rate)
    return amplitude * (2 * np.random.rand(n) - 1.0)
