import numpy as np
from wave_morph import morph_1d


# ---------------------------------------------------
# Swarm simulation & synth parameters
# ---------------------------------------------------

class SwarmParams:
    def __init__(
        self,
        num_agents=16,
        duration=4.0,
        sample_rate=44100,
        control_rate=200.0,
        base_freq=110.0,
        freq_spread_cents=20.0,   # random pitch offset per agent
        shape_a="sine",
        shape_b="saw",
        max_speed=2.0,
        neighbor_radius=3.0,
        separation_radius=0.7,
        alignment_weight=0.06,
        cohesion_weight=0.01,
        separation_weight=0.15,
        morph_speed_scale=0.15,
        morph_density_scale=-0.05,
        min_amp=0.05,
        max_amp=0.5,
        random_seed=0,
    ):
        self.num_agents = num_agents
        self.duration = duration
        self.sample_rate = sample_rate
        self.control_rate = control_rate
        self.base_freq = base_freq
        self.freq_spread_cents = freq_spread_cents
        self.shape_a = shape_a
        self.shape_b = shape_b

        self.max_speed = max_speed
        self.neighbor_radius = neighbor_radius
        self.separation_radius = separation_radius
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight
        self.separation_weight = separation_weight
        self.morph_speed_scale = morph_speed_scale
        self.morph_density_scale = morph_density_scale
        self.min_amp = min_amp
        self.max_amp = max_amp
        self.random_seed = random_seed


# ---------------------------------------------------
# Swarm simulation (control-rate: boid-like)
# ---------------------------------------------------

def simulate_swarm(params: SwarmParams):
    """
    Simulate swarm at control rate.
    Returns:
        freqs      : (num_agents,) base freq per agent
        pans       : (num_agents,) pan in [-1,1]
        morph_ctrl : (num_agents, n_ctrl) morph values [0,1]
        amp_ctrl   : (num_agents, n_ctrl) amplitudes [0,1]
    """
    rng = np.random.RandomState(params.random_seed)

    num_agents = params.num_agents
    duration = params.duration
    control_rate = params.control_rate

    n_ctrl = int(duration * control_rate)
    ctrl_times = np.linspace(0.0, duration, n_ctrl, endpoint=False)

    # Agent state: positions & velocities in 2D
    pos = rng.normal(scale=1.0, size=(num_agents, 2))
    vel = rng.normal(scale=0.2, size=(num_agents, 2))

    # Base freqs with random detune in cents
    detune_cents = rng.normal(scale=params.freq_spread_cents, size=num_agents)
    freq_ratios = 2 ** (detune_cents / 1200.0)
    freqs = params.base_freq * freq_ratios

    # Stereo pan per agent in [-1,1]
    pans = rng.uniform(-1.0, 1.0, size=num_agents)

    # Outputs over control time
    morph_ctrl = np.zeros((num_agents, n_ctrl), dtype=float)
    amp_ctrl = np.zeros((num_agents, n_ctrl), dtype=float)

    dt = 1.0 / control_rate

    for k in range(n_ctrl):
        # Pairwise distances (num_agents x num_agents)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # i - j
        dists = np.linalg.norm(diff, axis=2) + np.eye(num_agents)  # avoid zeros on diag

        # Neighbors within radius
        neighbor_mask = dists < params.neighbor_radius
        # Too-close neighbors
        close_mask = dists < params.separation_radius

        new_vel = vel.copy()

        for i in range(num_agents):
            neighbors = neighbor_mask[i]
            neighbors[i] = False  # ignore self
            if not np.any(neighbors):
                continue

            # Alignment: match neighbor velocity
            avg_vel = vel[neighbors].mean(axis=0)

            # Cohesion: steer toward neighbor center
            center = pos[neighbors].mean(axis=0)
            cohesion_vec = center - pos[i]

            # Separation: push away from very close neighbors
            close_neighbors = close_mask[i]
            close_neighbors[i] = False
            separation_vec = np.zeros(2)
            if np.any(close_neighbors):
                separation_vec = (pos[i] - pos[close_neighbors]).sum(axis=0)

            # Apply weighted sum
            new_vel[i] += (
                params.alignment_weight * (avg_vel - vel[i]) +
                params.cohesion_weight * cohesion_vec +
                params.separation_weight * separation_vec
            )

        # Update velocities
        vel = new_vel

        # Limit speed
        speeds = np.linalg.norm(vel, axis=1)
        too_fast = speeds > params.max_speed
        vel[too_fast] *= (params.max_speed / speeds[too_fast])[:, np.newaxis]

        # Integrate positions
        pos += vel * dt

        # Derive morph + amplitude from speed & local density
        speeds = np.linalg.norm(vel, axis=1)
        # Local density ~ number of neighbors
        neighbor_counts = neighbor_mask.sum(axis=1) - 1  # exclude self

        # Map speed & density to morph
        #   higher speed -> brighter (toward shape_b)
        #   higher density -> darker (toward shape_a)
        norm_speed = speeds / (params.max_speed + 1e-6)
        norm_density = neighbor_counts / max(1, num_agents - 1)

        morph = 0.5 + params.morph_speed_scale * norm_speed + params.morph_density_scale * norm_density
        morph = np.clip(morph, 0.0, 1.0)

        # Amplitude: louder in medium density, quieter when crowded or isolated
        density_centered = np.abs(norm_density - 0.5)  # 0 at medium density
        amp = params.max_amp - density_centered * (params.max_amp - params.min_amp)
        amp = np.clip(amp, params.min_amp, params.max_amp)

        morph_ctrl[:, k] = morph
        amp_ctrl[:, k] = amp

    return freqs, pans, ctrl_times, morph_ctrl, amp_ctrl


# ---------------------------------------------------
# Render swarm to stereo audio
# ---------------------------------------------------

def render_swarm(params: SwarmParams):
    """
    High-level call:
      - simulates swarm at control rate
      - interpolates morph/amp to audio rate
      - renders morphing oscillators for each agent
      - mixes to stereo
    Returns:
      audio_stereo: shape (n_samples, 2)
    """
    sr = params.sample_rate
    duration = params.duration
    n_samples = int(duration * sr)
    t_audio = np.arange(n_samples) / sr

    freqs, pans, ctrl_times, morph_ctrl, amp_ctrl = simulate_swarm(params)

    num_agents = params.num_agents

    # Interpolate control signals up to audio rate
    morph_audio = np.zeros((num_agents, n_samples), dtype=float)
    amp_audio = np.zeros((num_agents, n_samples), dtype=float)

    for i in range(num_agents):
        morph_audio[i] = np.interp(t_audio, ctrl_times, morph_ctrl[i])
        amp_audio[i] = np.interp(t_audio, ctrl_times, amp_ctrl[i])

    # Render each agent as a morphing oscillator and pan it
    left = np.zeros(n_samples, dtype=float)
    right = np.zeros(n_samples, dtype=float)

    for i in range(num_agents):
        f = freqs[i]
        m = morph_audio[i]
        a = amp_audio[i]
        pan = pans[i]  # -1 = left, +1 = right

        # Render morphing wave (sine->shape_b or whatever you set)
        agent_wave = morph_1d(
            freq=f,
            duration=duration,
            morph=m,
            shape_a=params.shape_a,
            shape_b=params.shape_b,
            sample_rate=sr,
            amplitude=1.0,  # we'll apply amp separately
        )

        # Apply amplitude envelope
        agent_wave *= a

        # Simple linear pan
        # Convert pan [-1,1] to gains
        g_l = np.sqrt(0.5 * (1.0 - pan))
        g_r = np.sqrt(0.5 * (1.0 + pan))

        left += agent_wave * g_l
        right += agent_wave * g_r

    # Stack stereo
    audio_stereo = np.stack([left, right], axis=-1)

    # Optional soft normalization to avoid clipping
    peak = np.max(np.abs(audio_stereo))
    if peak > 1.0:
        audio_stereo /= peak

    return audio_stereo


# ---------------------------------------------------
# Example usage
# ---------------------------------------------------

if __name__ == "__main__":
    # Basic swarming pad
    params = SwarmParams(
        num_agents=24,
        duration=6.0,
        sample_rate=44100,
        control_rate=150.0,
        base_freq=110.0,
        freq_spread_cents=15.0,
        shape_a="sine",
        shape_b="saw",       # sine ↔ saw morph per agent
        max_speed=2.0,
        neighbor_radius=3.0,
        separation_radius=0.7,
        alignment_weight=0.06,
        cohesion_weight=0.01,
        separation_weight=0.15,
        morph_speed_scale=0.25,
        morph_density_scale=-0.10,
        min_amp=0.03,
        max_amp=0.4,
        random_seed=1,
    )

    audio = render_swarm(params)

    print("Rendered swarm audio:", audio.shape)
    print("Peak level:", np.max(np.abs(audio)))
