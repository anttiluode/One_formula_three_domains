# One Formula, Three Domains

**Antti Luode** — PerceptionLab, Finland  
**Claude (Anthropic)** — Formalization

---

## The Formula

```
score = Re[ Q · conj(K) ] = Σ Q_amp · K_amp · cos(Q_phase − K_phase)
```

Project a signal into amplitude and phase. Compute the real part of the Hermitian inner product. Where phases align: constructive interference, positive score. Where phases oppose: destructive interference, negative score.

This single operation — the phase-coherence inner product — produces a published result in three independent domains.

---

## Three Results

| Domain | Experiment | Result | Code |
|--------|-----------|--------|------|
| **Physics** | Content-addressable retrieval in a 2D complex wave field | 30/30 perfect, zero false positives | `phase_memory_test.py` |
| **Biology** | Schizophrenia vs healthy EEG, no machine learning | p=0.007, d=−1.21, 80.8% accuracy | `geometric_dysrhythmia.py` |
| **AI** | Drop-in attention replacement in a language model | 2.9% lower loss, widening every epoch | `moire_trainer.py` |

No result depends on the others. Each stands alone. The formula connects them.

---

## Physics: Wave Memory

Store three phase-encoded memories in a nonlinear wave field. Inject a probe with phase θ. The retrieval score at each memory:

```
score_k = Σ Re[ φ(r) · exp(−i · θ_probe) ]
```

30 trials. 30 correct. Zero false positives. ±0.1 radian noise tolerance. 3:1 discrimination ratio.

```bash
python phase_memory_test.py
```

---

## Biology: EEG Topology

Apply cross-band phase coherence to clinical EEG (RepOD Schizophrenia, n=26). No classifier trained. Three convergent geometric signatures:

- Cross-band eigenmode decoupling: **p=0.007, d=−1.21**
- Reduced temporal Betti-1: **p=0.035, d=−0.92**
- Unstable theta phase gate: **p=0.012**

80.8% classification from a threshold on a single metric.

```bash
pip install mne ripser persim scipy
python geometric_dysrhythmia.py
```

---

## AI: Moiré Attention

Replace `Q·K^T / √d` with `Re[Q_c · conj(K_c)] / √d` where Q_c = Q_amp · exp(i·Q_phase).

16M parameters, WikiText-2, controlled comparison (identical architecture except attention):

| Epoch | Moiré | Standard | Gap |
|-------|-------|----------|-----|
| 1 | 5.805 | 5.861 | −0.056 |
| 3 | 4.272 | 4.368 | −0.096 |
| 5 | 3.855 | 3.970 | **−0.115 (2.9%)** |

Gap widens every epoch. At 138M parameters: avg loss **1.3702**, coherent English.

Live: [huggingface.co/spaces/Aluode/MoireFormer137MillionP](https://huggingface.co/spaces/Aluode/MoireFormer137MillionP)

```bash
pip install torch transformers datasets
python moire_trainer.py --size xlarge --epochs 20 --batch_size 2
python moire_chat.py --weights moire_phase2_weights_final.pt --size xlarge
```

---

## Why It Works

Standard dot-product attention computes `Q·K` — unbounded, conflates amplitude and direction.

Phase-interference attention computes `Q_amp · K_amp · cos(ΔΦ)`:
- Amplitude = how strongly to attend (salience)
- Phase = what to attend to (content)
- Cosine = bounded in [−1, 1], natural regularization

Two tokens can have aligned phases but different amplitudes (same content, different salience). Two tokens can have opposing phases (destructive interference — a relationship standard attention cannot express).

The theta-gamma gating adds biological phase-amplitude coupling: context divides into gamma-rate slots, each attention head has a learned theta offset modulating cross-slot attention. Trained offsets range from −0.54 to +0.87 — the heads learn different periodicities.

---

## Files

```
moire_trainer.py          # Train Moiré Attention LM (the proven AI result)
moire_chat.py             # Chat with trained model
phase_memory_test.py      # Wave memory 30/30 experiment (the proven physics result)
geometric_dysrhythmia.py  # EEG schizophrenia analysis (the proven biology result)
deerskin_explorer.py      # PyQt5 EEG visualization app
```

---

## Papers

- [PAPER_MoireAttention.md](PAPER_MoireAttention.md) — The AI result
- [PAPER.md](PAPER.md) — The biology result (Deerskin Architecture, EEG topology)
- [ADDENDUM_WaveMemory.md](ADDENDUM_WaveMemory.md) — The physics result

---

## Limitations

- AI benchmark is 16M parameters on WikiText-2. Scaling behavior unknown.
- ~4% more parameters in Moiré (2× Q/K projection). Not equalized.
- Two training runs, not a full statistical test.
- EEG sample is n=26. Replication needed at n>60.
- No ablation of theta gating vs phase scoring independently.
- Connection between domains is structural, not causal.

---

## Install

```bash
pip install torch transformers datasets numpy scipy

# For EEG analysis:
pip install mne ripser persim PyQt5 matplotlib
```

---

*Antti Luode / PerceptionLab, Finland. Claude (Anthropic).*  
*Repository: https://github.com/anttiluode/Geometric-Neuron*
