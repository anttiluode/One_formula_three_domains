"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CLOCKFIELD DNA MEMORY — Phase-Encoded Soliton Crystals                     ║
║                                                                              ║
║  The Breakthrough:                                                           ║
║  Information cannot be stored in the *Amplitude* of a field, because         ║
║  amplitude equates to Mass/Energy. The Mexican Hat potential forces all      ║
║  structures to normalize to the exact same equilibrium mass.                 ║
║                                                                              ║
║  The Solution (The DNA Analogy):                                             ║
║  1. THE BACKBONE (Amplitude): All solitons are injected at the exact same    ║
║     amplitude (2.0). The field happily preserves this uniform structure.     ║
║  2. THE GENES (Phase): The actual ASCII data is stored in the Phase Angle    ║
║     of the complex field (from -1.5 to 1.5 radians). It costs zero energy    ║
║     to hold a phase, so the Clockfield freezes it perfectly.                 ║
║  3. THE LATTICE (Structural Repulsion): We add an alternating π phase        ║
║     between every letter. This guarantees destructive interference at the    ║
║     borders, creating "fast-time moats" that prevent the letters from        ║
║     ever merging.                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
from pathlib import Path

# ═════════════════════════════════════════════════════════════════════════════
# 1. THE COMPLEX FIELD
# ═════════════════════════════════════════════════════════════════════════════

class ClockfieldComplexMemory:
    def __init__(self, N=2048, dt=0.04, damping=0.003,
                 tension=10.0, pot_lin=1.0, pot_cub=0.25,
                 biharmonic=0.0003):
        self.N = N
        self.dt = dt
        self.damping = damping
        self.tension = tension
        self.pot_lin = pot_lin
        self.pot_cub = pot_cub
        self.biharmonic = biharmonic

        # Field state is COMPLEX
        self.phi = np.zeros(N, dtype=np.complex128)
        self.phi_old = np.zeros_like(self.phi)
        self.t = 0.0
        self.step_count = 0

        self.memories = []  

    def _laplacian(self, f):
        return np.roll(f, 1) + np.roll(f, -1) - 2 * f

    def _biharmonic(self, f):
        lap = self._laplacian(f)
        return self._laplacian(lap)

    def step(self, n_steps=1):
        for _ in range(n_steps):
            lap = self._laplacian(self.phi)
            biharm = self._biharmonic(self.phi)

            # Clockfield Metric: Driven strictly by Amplitude (Mass)
            phi_sq = np.abs(self.phi)**2
            c2 = 1.0 / (1.0 + self.tension * phi_sq + 1e-9)

            # Mexican hat potential: Preserves Amplitude, Ignores Phase
            Vp = -self.pot_lin * self.phi + self.pot_cub * phi_sq * self.phi

            # Acceleration
            acc = c2 * lap - Vp - self.biharmonic * biharm

            # Velocity
            vel = self.phi - self.phi_old

            # Update
            phi_new = self.phi + (1.0 - self.damping * self.dt) * vel + self.dt**2 * acc

            self.phi_old = self.phi.copy()
            self.phi = phi_new
            self.t += self.dt
            self.step_count += 1

    def inject_memory(self, location, amplitude, phase, width=15, label="", index=0):
        """
        Inject a pulse with fixed amplitude and a specific PHASE angle.
        """
        x = np.arange(self.N)
        # Construct the complex soliton
        pulse = amplitude * np.exp(-(x - location)**2 / (2 * width**2)) * np.exp(1j * phase)
        
        self.phi += pulse
        self.phi_old += pulse  
        self.memories.append({
            'index': index,
            'location': location,
            'target_amp': amplitude,
            'injected_phase': phase,
            'width': width,
            'label': label,
        })

    def read_memory(self, location, width=15):
        """Read both the backbone Amplitude and the genetic Phase."""
        window_slice = self.phi[max(0, location-width):min(self.N, location+width)]
        if len(window_slice) == 0: return 0.0, 0.0
        
        # Find the exact peak of the soliton to read its pure phase
        peak_idx = np.argmax(np.abs(window_slice))
        peak_val = window_slice[peak_idx]
        
        return np.abs(peak_val), np.angle(peak_val)

    def read_all_memories(self):
        results = []
        for mem in self.memories:
            amp, phase = self.read_memory(mem['location'], mem['width'])
            results.append({
                'label': mem['label'],
                'index': mem['index'],
                'location': mem['location'],
                'readback_amp': amp,
                'readback_phase': phase,
            })
        return results

    def compute_gamma_field(self):
        return 1.0 / np.sqrt(1.0 + self.tension * np.abs(self.phi)**2 + 1e-9)


# ═════════════════════════════════════════════════════════════════════════════
# 2. DATA ENCODING / DECODING (THE GENETICS)
# ═════════════════════════════════════════════════════════════════════════════

def encode_chars_to_phases(text):
    """Map ASCII characters to a phase angle between -1.5 and 1.5 radians."""
    phases = []
    for c in text:
        # ASCII printable range: 32 (space) to 126 (~)
        val = np.clip(ord(c), 32, 126)
        normalized = (val - 32) / 94.0  # 0.0 to 1.0
        phase = (normalized * 3.0) - 1.5  # -1.5 to 1.5
        phases.append(phase)
    return phases

def decode_phases_to_chars(phases):
    """Map phase angles back to ASCII characters."""
    chars = []
    for p in phases:
        normalized = (p + 1.5) / 3.0
        val = int(round(normalized * 94.0 + 32))
        val = max(32, min(126, val))
        chars.append(chr(val))
    return "".join(chars)


# ═════════════════════════════════════════════════════════════════════════════
# 3. THE EXPERIMENT
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment():
    RESULTS_DIR = Path("./wave_memory_results")
    RESULTS_DIR.mkdir(exist_ok=True)

    field = ClockfieldComplexMemory(
        N=2048, dt=0.04, damping=0.003,
        tension=10.0, pot_lin=1.0, pot_cub=0.25, # Target Amplitude = sqrt(1.0/0.25) = 2.0
        biharmonic=0.0003,
    )

    word1 = "CLOCKFLD"
    word2 = "DEERSKIN"

    data_phases1 = encode_chars_to_phases(word1)
    data_phases2 = encode_chars_to_phases(word2)

    # Tight packing! The alternating structural phase will protect them.
    spacing = 50   
    start1 = 300
    start2 = 1300
    locs1 = [start1 + i * spacing for i in range(len(word1))]
    locs2 = [start2 + i * spacing for i in range(len(word2))]

    PHASE1_STEPS = 0
    PHASE2_STEPS = 2500
    PHASE3_STEPS = 2500
    PHASE4_STEPS = 2500
    STEPS_PER_FRAME = 25

    # ── Setup figure ─────────────────────────────────────────────────────
    plt.ion()
    fig = plt.figure(figsize=(16, 10), facecolor='#0a0a12')
    fig.suptitle('CLOCKFIELD DNA MEMORY (Phase-Encoded Solitons)', fontsize=16, color='#00ccff', fontweight='bold')
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3, left=0.06, right=0.96, top=0.92, bottom=0.06)

    ax_amp    = fig.add_subplot(gs[0, 0])
    ax_phase  = fig.add_subplot(gs[0, 1])
    ax_gamma  = fig.add_subplot(gs[1, :])
    ax_history= fig.add_subplot(gs[2, 0])
    ax_text   = fig.add_subplot(gs[2, 1])

    for ax in [ax_amp, ax_phase, ax_gamma, ax_history, ax_text]:
        ax.set_facecolor('#0a0a12')
        ax.tick_params(colors='#4a5868', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#1a2030')

    history1 = []
    history2 = []

    # ── Phase 1: Inject Word 1 ────────────────────────────────────────────
    for i, (loc, d_phase) in enumerate(zip(locs1, data_phases1)):
        # The secret to the crystal: Alternating structural phase keeps them separated
        structural_phase = (i % 2) * np.pi 
        total_phase = d_phase + structural_phase
        field.inject_memory(loc, amplitude=2.0, phase=total_phase, width=12, label=word1[i], index=i)

    # ── Phase 2: Crystallize Word 1 ───────────────────────────────────────
    steps_done = 0
    while steps_done < PHASE2_STEPS:
        field.step(STEPS_PER_FRAME)
        steps_done += STEPS_PER_FRAME
        _record_history(field, len(word1), history1, history2)
        if steps_done % 100 == 0:
            _update_display(fig, ax_amp, ax_phase, ax_gamma, ax_history, ax_text, 
                            field, word1, word2, locs1, locs2, history1, history2, "Phase 2: Crystallizing Word 1")

    # ── Phase 3: Inject Word 2 ────────────────────────────────────────────
    for i, (loc, d_phase) in enumerate(zip(locs2, data_phases2)):
        structural_phase = (i % 2) * np.pi 
        total_phase = d_phase + structural_phase
        field.inject_memory(loc, amplitude=2.0, phase=total_phase, width=12, label=word2[i], index=i)

    # ── Phase 4: Evolve Both ──────────────────────────────────────────────
    steps_done = 0
    while steps_done < PHASE3_STEPS + PHASE4_STEPS:
        field.step(STEPS_PER_FRAME)
        steps_done += STEPS_PER_FRAME
        _record_history(field, len(word1), history1, history2)
        if steps_done % 100 == 0:
            _update_display(fig, ax_amp, ax_phase, ax_gamma, ax_history, ax_text, 
                            field, word1, word2, locs1, locs2, history1, history2, "Phase 4: Field Equilibration")

    # ── Final Readout ─────────────────────────────────────────────────────
    reads = field.read_all_memories()
    
    # Decode Word 1
    w1_reads = reads[:len(word1)]
    w1_data_phases = []
    for r in w1_reads:
        structural_phase = (r['index'] % 2) * np.pi
        raw_phase = r['readback_phase']
        # Subtract structural phase and wrap back to [-pi, pi]
        data_phase = (raw_phase - structural_phase + np.pi) % (2 * np.pi) - np.pi
        w1_data_phases.append(data_phase)
    dec1 = decode_phases_to_chars(w1_data_phases)

    # Decode Word 2
    w2_reads = reads[len(word1):]
    w2_data_phases = []
    for r in w2_reads:
        structural_phase = (r['index'] % 2) * np.pi
        raw_phase = r['readback_phase']
        data_phase = (raw_phase - structural_phase + np.pi) % (2 * np.pi) - np.pi
        w2_data_phases.append(data_phase)
    dec2 = decode_phases_to_chars(w2_data_phases)

    print("\n" + "="*60)
    print("FINAL DECODE RESULTS")
    print("="*60)
    print(f"Word 1 Target: {word1}")
    print(f"Word 1 Actual: {dec1}")
    print(f"Word 2 Target: {word2}")
    print(f"Word 2 Actual: {dec2}")
    if word1 == dec1 and word2 == dec2:
        print("\n★ SUCCESS! The field perfectly preserved the DNA data. ★")
    print("="*60)

    _update_display(fig, ax_amp, ax_phase, ax_gamma, ax_history, ax_text, 
                    field, word1, word2, locs1, locs2, history1, history2, "FINAL STATE", dec1, dec2)

    plt.ioff()
    plt.show()

def _record_history(field, w1_len, h1, h2):
    reads = field.read_all_memories()
    
    dp1 = []
    for r in reads[:w1_len]:
        sp = (r['index'] % 2) * np.pi
        dp = (r['readback_phase'] - sp + np.pi) % (2 * np.pi) - np.pi
        dp1.append(dp)
    h1.append((field.t, dp1))

    if len(reads) > w1_len:
        dp2 = []
        for r in reads[w1_len:]:
            sp = (r['index'] % 2) * np.pi
            dp = (r['readback_phase'] - sp + np.pi) % (2 * np.pi) - np.pi
            dp2.append(dp)
        h2.append((field.t, dp2))

def _update_display(fig, ax_amp, ax_phase, ax_gamma, ax_history, ax_text, 
                    field, word1, word2, locs1, locs2, rh1, rh2, title, dec1="", dec2=""):
    x = np.arange(field.N)
    abs_phi = np.abs(field.phi)
    ang_phi = np.angle(field.phi)

    # 1. Amplitude (The Backbone)
    ax_amp.clear()
    ax_amp.plot(x, abs_phi, color='#ffffff', linewidth=1.5)
    ax_amp.set_title('Amplitude |φ| (The DNA Backbone)', color='#8899aa', fontsize=10)
    ax_amp.set_xlim(0, field.N)
    ax_amp.set_ylim(0, 2.5)
    ax_amp.set_facecolor('#0a0a12')

    # 2. Phase (The Genes)
    ax_phase.clear()
    # Only show phase where amplitude is significant to reduce visual noise
    mask = abs_phi > 0.5
    ax_phase.scatter(x[mask], ang_phi[mask], color='#00ccff', s=2, alpha=0.5)
    ax_phase.set_title('Phase Angle θ (The Genetic Data)', color='#8899aa', fontsize=10)
    ax_phase.set_xlim(0, field.N)
    ax_phase.set_ylim(-np.pi, np.pi)
    ax_phase.set_facecolor('#0a0a12')

    # 3. Gamma
    ax_gamma.clear()
    gamma = field.compute_gamma_field()
    ax_gamma.fill_between(x, gamma, 1.0, color='#cc44ff', alpha=0.3)
    ax_gamma.plot(x, gamma, color='#cc44ff', linewidth=1)
    ax_gamma.set_title('Clockfield Metric Γ (Protected Temporal Wells)', color='#8899aa', fontsize=10)
    ax_gamma.set_xlim(0, field.N)
    ax_gamma.set_ylim(0, 1.05)
    ax_gamma.set_facecolor('#0a0a12')

    # 4. History
    ax_history.clear()
    if rh1:
        times = [t for t, _ in rh1]
        for i in range(len(rh1[0][1])):
            ax_history.plot(times, [r[i] for _, r in rh1], linewidth=1)
    if rh2:
        times = [t for t, _ in rh2]
        for i in range(len(rh2[0][1])):
            ax_history.plot(times, [r[i] for _, r in rh2], linewidth=1, linestyle='--')
    
    ax_history.set_title('Decoded Data Phase Stability Over Time', color='#8899aa', fontsize=10)
    ax_history.set_ylim(-2.0, 2.0)
    ax_history.set_facecolor('#0a0a12')

    # 5. Text
    ax_text.clear()
    ax_text.axis('off')
    ax_text.set_facecolor('#0a0a12')
    
    t_lines = [
        f"TIME: {field.t:.1f} | STEPS: {field.step_count}",
        f"STATUS: {title}",
        f"",
    ]
    if dec1:
        t_lines += [f"WORD 1 TARGET: {word1}", f"WORD 1 ACTUAL: {dec1}", f""]
    if dec2:
        t_lines += [f"WORD 2 TARGET: {word2}", f"WORD 2 ACTUAL: {dec2}"]

    for i, line in enumerate(t_lines):
        ax_text.text(0.05, 0.9 - i*0.1, line, color='#aabbcc', fontsize=11, fontfamily='monospace', fontweight='bold')

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.01)

if __name__ == "__main__":
    run_experiment()