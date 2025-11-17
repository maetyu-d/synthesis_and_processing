import numpy as np
from waveforms import sine_wave


# -------------------------------
# Basic gain / drive helper
# -------------------------------
def apply_drive(x, drive):
    """
    Simple pre-gain to push the signal into the nonlinearities.
    drive > 1.0 increases distortion.
    """
    return x * drive


# -------------------------------
# 1) Hard clip
# -------------------------------
def hard_clip(x, threshold=0.8):
    """
    Hard clipping at ±threshold.
    """
    return np.clip(x, -threshold, +threshold)


# -------------------------------
# 2) Soft clip (tanh)
# -------------------------------
def soft_clip_tanh(x, drive=1.0):
    """
    Classic soft clip using tanh.
    drive controls how hard we hit the nonlinearity.
    """
    x = apply_drive(x, drive)
    return np.tanh(x)


# -------------------------------
# 3) Polynomial waveshaper
#    (odd-order distortion)
# -------------------------------
def poly_waveshape(x, drive=1.0, k3=0.5, k5=0.2):
    """
    Simple odd-order polynomial waveshaper:
    y = x + k3 * x^3 + k5 * x^5, with pre-drive.
    """
    x = apply_drive(x, drive)
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    y = x + k3 * x3 + k5 * x5
    # optional normalization / safety clip
    return np.clip(y, -1.5, 1.5)


# -------------------------------
# 4) Chebyshev waveshape
#    (spectral “designer”)
# -------------------------------
def chebyshev_waveshape(x, order=3, drive=1.0):
    """
    Chebyshev-based waveshaper.
    For |x| ≤ 1: T_n(x) = cos(n * arccos(x))
    This emphasizes specific harmonics.
    """
    x = apply_drive(x, drive)
    # keep in [-1,1] so arccos is defined
    x = np.clip(x, -1.0, 1.0)
    return np.cos(order * np.arccos(x))


# -------------------------------
# 5) Simple wavefolder
#    (triangle-style folding)
# -------------------------------
def wavefolder_triangle(x, drive=5.0, fold_level=1.0):
    """
    Simple triangle wavefolder.

    1. apply drive to push the signal past ±fold_level
    2. fold it back using a triangle mapping

    This is not a model of a specific analog module, but a handy
    digital wavefolder.
    """
    x = apply_drive(x, drive)

    # scale so fold_level is our 'boundary'
    a = fold_level
    # map x into a repeating triangle between -a and +a

    # Shift so 0 is at -a, repeat every 4a:
    #   - range-wrap using modulo
    y = np.mod(x + a, 4 * a)

    # Reflect into triangle shape between 0 and 2a
    y = np.where(y > 2 * a, 4 * a - y, y)

    # Shift down to be centered around 0
    y = y - a

    # optional safety clip
    return np.clip(y, -a, +a)


# -------------------------------
# Example: generate & shape a sine
# -------------------------------
if __name__ == "__main__":
    sr = 44100
    duration = 1.0
    freq = 110.0

    # 1) Base sine from waveforms.py
    base = sine_wave(freq, duration, sample_rate=sr, amplitude=1.0)

    # 2) Hard-clipped sine
    hard = hard_clip(base, threshold=0.6)

    # 3) Soft-clipped sine
    soft = soft_clip_tanh(base, drive=3.0)

    # 4) Polynomial-shaped sine
    poly = poly_waveshape(base, drive=2.0, k3=0.7, k5=0.3)

    # 5) Chebyshev-shaped sine (3rd-order: strong 3rd harmonic)
    cheb3 = chebyshev_waveshape(base, order=3, drive=1.5)

    # 6) Wavefolded sine (triangle wavefolder)
    folded = wavefolder_triangle(base, drive=6.0, fold_level=0.8)

    print("Generated:")
    print(" base:", base.shape)
    print(" hard:", hard.shape)
    print(" soft:", soft.shape)
    print(" poly:", poly.shape)
    print(" cheb3:", cheb3.shape)
    print(" folded:", folded.shape)
