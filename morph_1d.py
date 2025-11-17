import numpy as np
from waveforms import sine_wave, triangle_wave, sawtooth_wave, square_wave, pulse_wave, noise_wave


# ---------------------------------------------------
# Wave generator lookup
# ---------------------------------------------------

WAVE_GENERATORS = {
    "sine":     sine_wave,
    "triangle": triangle_wave,
    "tri":      triangle_wave,
    "saw":      sawtooth_wave,   # rising saw by default
    "square":   square_wave,
    "pulse":    pulse_wave,
    "noise":    lambda f, d, sample_rate=44100, amplitude=1.0: noise_wave(d, sample_rate, amplitude),
}


def make_wave(shape, freq, duration, sample_rate=44100, amplitude=1.0, phase=0.0, **kwargs):
    """
    Create one waveform by name.
    """
    shape = shape.lower()
    if shape not in WAVE_GENERATORS:
        raise ValueError(f"Unknown shape '{shape}'. Available: {list(WAVE_GENERATORS.keys())}")

    gen = WAVE_GENERATORS[shape]

    if shape == "saw" and "rising" not in kwargs:
        kwargs["rising"] = True

    if shape == "pulse" and "duty" not in kwargs:
        kwargs["duty"] = 0.5

    return gen(freq, duration, sample_rate=sample_rate, amplitude=amplitude, phase=phase, **kwargs)


# ---------------------------------------------------
# 1D morphing: supports scalar OR time-varying morph
# ---------------------------------------------------

def morph_1d(
    freq,
    duration,
    morph,                     # scalar or array, 0..1
    shape_a="sine",
    shape_b="saw",
    sample_rate=44100,
    amplitude=1.0,
    phase=0.0,
    **kwargs
):
    """
    Morph between two wave shapes.

    morph:
      - scalar (float): 0 → shape_a, 1 → shape_b (static)
      - array (len = num_samples): time-varying 0..1
    """
    wa = make_wave(shape_a, freq, duration, sample_rate,
                   amplitude=amplitude, phase=phase, **kwargs)
    wb = make_wave(shape_b, freq, duration, sample_rate,
                   amplitude=amplitude, phase=phase, **kwargs)

    n = len(wa)
    m = np.asarray(morph, dtype=float)

    if m.ndim == 0:
        # static morph → broadcast
        m = np.full(n, np.clip(m, 0.0, 1.0))
    else:
        if len(m) != n:
            raise ValueError(f"morph array length {len(m)} != num samples {n}")
        m = np.clip(m, 0.0, 1.0)

    y = (1.0 - m) * wa + m * wb
    return y


# ---------------------------------------------------
# LFO-driven morph (time-varying)
# ---------------------------------------------------

def morph_1d_lfo(
    freq,
    duration,
    shape_a="sine",
    shape_b="saw",
    sample_rate=44100,
    amplitude=1.0,
    phase=0.0,
    lfo_rate_hz=0.5,      # Hz
    lfo_depth=1.0,        # 0..1
    lfo_center=0.5        # 0..1
):
    """
    Time-varying morph driven by a sine LFO.

    lfo_center = 0.5, depth = 0.5 → morph swings 0..1
    lfo_center = 0.5, depth = 0.25 → morph swings 0.25..0.75
    """
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate

    # raw LFO in [-1, 1]
    lfo = np.sin(2 * np.pi * lfo_rate_hz * t)

    # map to around lfo_center
    morph = lfo_center + lfo_depth * lfo

    return morph_1d(
        freq=freq,
        duration=duration,
        morph=morph,
        shape_a=shape_a,
        shape_b=shape_b,
        sample_rate=sample_rate,
        amplitude=amplitude,
        phase=phase,
    )


# ---------------------------------------------------
# Envelope-driven morph (simple attack-decay)
# ---------------------------------------------------

def linear_attack_decay_env(n_samples, attack_frac=0.2):
    """
    Simple attack-decay envelope in [0,1].
    attack_frac: fraction of duration spent going 0→1 (attack),
                 remainder is 1→0 (decay).
    """
    attack_samples = int(n_samples * attack_frac)
    decay_samples = n_samples - attack_samples

    attack = np.linspace(0.0, 1.0, attack_samples, endpoint=False)
    decay = np.linspace(1.0, 0.0, decay_samples, endpoint=True)

    return np.concatenate([attack, decay])


def morph_1d_envelope(
    freq,
    duration,
    shape_a="sine",
    shape_b="square",
    sample_rate=44100,
    amplitude=1.0,
    phase=0.0,
    attack_frac=0.2
):
    """
    Time-varying morph driven by an attack-decay envelope.
    Starts at shape_a, moves toward shape_b, then back.
    """
    n = int(duration * sample_rate)
    env = linear_attack_decay_env(n, attack_frac=attack_frac)

    return morph_1d(
        freq=freq,
        duration=duration,
        morph=env,
        shape_a=shape_a,
        shape_b=shape_b,
        sample_rate=sample_rate,
        amplitude=amplitude,
        phase=phase,
    )


# ---------------------------------------------------
# Example usage
# ---------------------------------------------------

if __name__ == "__main__":
    sr = 44100
    dur = 4.0
    f = 110.0

    # 1) Static morph (0.25 between sine and saw)
    y_static = morph_1d(
        freq=f,
        duration=dur,
        morph=0.25,
        shape_a="sine",
        shape_b="saw",
        sample_rate=sr,
        amplitude=0.8,
    )

    # 2) LFO morph: sine ↔ saw over time
    y_lfo = morph_1d_lfo(
        freq=f,
        duration=dur,
        shape_a="sine",
        shape_b="saw",
        sample_rate=sr,
        amplitude=0.8,
        lfo_rate_hz=0.25,   # one full sweep every 4 seconds
        lfo_depth=0.5,      # full 0..1 swing (center=0.5)
        lfo_center=0.5
    )

    # 3) Envelope morph: gesture sine → square → sine
    y_env = morph_1d_envelope(
        freq=f,
        duration=dur,
        shape_a="sine",
        shape_b="square",
        sample_rate=sr,
        amplitude=0.8,
        attack_frac=0.3
    )

    print("Static morph samples:", len(y_static))
    print("LFO morph samples:   ", len(y_lfo))
    print("Env morph samples:   ", len(y_env))
