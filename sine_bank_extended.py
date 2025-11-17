import numpy as np
from waveforms import sine_wave


# ----------------------------
# Cents ↔ ratio helpers
# ----------------------------
def cents_to_ratio(cents):
    return 2 ** (cents / 1200.0)


# ----------------------------
# Sine bank with detune, supersaw (I know it's a sine, but what else to call it) spread,
# Gaussian drift, and LFO detune modulation
# ----------------------------
def sine_bank(
    frequencies,
    duration,
    sample_rate=44100,
    amplitudes=None,
    detune=None,                 # base detune per oscillator (cents)
    supersaw_voices=1,           # number of voices per oscillator
    supersaw_spread_cents=0.0,   # total detune span between lowest & highest voice
    drift_std_cents=0.0,         # Gaussian-distributed static drift per voice
    lfo_rate=0.0,                # LFO rate in Hz (0 = off)
    lfo_depth_cents=0.0          # LFO depth in cents (peak)
):
    """
    frequencies: list of base frequencies in Hz
    amplitudes: list of amplitudes (same length as frequencies)
    detune:     None, scalar, or list of cents per base frequency
                (positive = sharper, negative = flatter)

    supersaw_voices: how many voices per base frequency
    supersaw_spread_cents: total spread across the supersaw voices (in cents)
                           e.g. 20 -> voices from about -10c to +10c

    drift_std_cents: std-dev (in cents) for Gaussian-distributed drift
                     per *voice* (static offset, different for each voice)

    lfo_rate: LFO frequency in Hz (0 disables LFO)
    lfo_depth_cents: peak LFO modulation in cents
    """

    num_osc = len(frequencies)
    if amplitudes is None:
        amplitudes = [1.0] * num_osc

    # Base detune handling
    if detune is None:
        detune = [0.0] * num_osc
    elif isinstance(detune, (int, float)):
        detune = [float(detune)] * num_osc
    else:
        detune = [float(d) for d in detune]

    # Time base
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate

    # LFO in cents over time (shared across voices for now)
    if lfo_rate > 0.0 and lfo_depth_cents != 0.0:
        lfo = np.sin(2 * np.pi * lfo_rate * t)  # [-1, 1]
        lfo_cents = lfo * lfo_depth_cents
    else:
        lfo_cents = None

    # Supersaw voice offsets in cents (static pattern)
    if supersaw_voices <= 1 or supersaw_spread_cents == 0.0:
        supersaw_offsets = np.array([0.0])
        supersaw_voices = 1
    else:
        # Spread voices evenly from -spread/2 to +spread/2
        supersaw_offsets = np.linspace(
            -supersaw_spread_cents / 2.0,
            +supersaw_spread_cents / 2.0,
            supersaw_voices
        )

    # Output buffer
    output = np.zeros(n_samples, dtype=float)

    # Build all oscillators
    for f_base, a_base, d_base in zip(frequencies, amplitudes, detune):
        for offset in supersaw_offsets:
            # Static drift per *voice* (Gaussian)
            if drift_std_cents > 0.0:
                drift = np.random.normal(loc=0.0, scale=drift_std_cents)
            else:
                drift = 0.0

            static_cents = d_base + offset + drift

            # If there's no time-varying LFO, we can use the static sine_wave
            if lfo_cents is None:
                f_detuned = f_base * cents_to_ratio(static_cents)
                output += a_base * sine_wave(
                    f_detuned, duration, sample_rate, amplitude=1.0
                )
            else:
                # Time-varying detune: static + LFO in cents
                cents_total = static_cents + lfo_cents         # shape (n_samples,)
                ratio_t = cents_to_ratio(cents_total)          # elementwise
                freq_t = f_base * ratio_t                      # instantaneous freq
                # Integrate phase: φ[n] = φ[n-1] + 2π * f[n]/Fs
                phase = 2 * np.pi * np.cumsum(freq_t / sample_rate)
                output += a_base * np.sin(phase)

    return output


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    sr = 44100
    duration = 2.0

    # One base note: A2
    base_freq = 110.0
    freqs = [base_freq]
    amps = [1.0]

    y = sine_bank(
        freqs,
        duration,
        sample_rate=sr,
        amplitudes=amps,

        # base detune in cents (on the core oscillator)
        detune=0.0,

        # supersaw "fatness"
        supersaw_voices=7,
        supersaw_spread_cents=30.0,   # ~±15 cents across voices

        # static drift per voice (Gaussian)
        drift_std_cents=3.0,          # subtle organic offset

        # LFO modulating detune
        lfo_rate=0.3,                 # very slow wobble
        lfo_depth_cents=8.0           # ±8 cents around static detune
    )

    print("Generated samples:", len(y))
    print("Peak amplitude:", np.max(np.abs(y)))
