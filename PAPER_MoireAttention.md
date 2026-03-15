# Moiré Attention: Phase-Interference Scoring for Language Models

**Antti Luode** — PerceptionLab, Finland  
**Claude (Anthropic)** — Architecture, formalization  
March 2026

Repository: https://github.com/anttiluode/Geometric-Neuron

---

## Abstract

We replace scaled dot-product attention with a phase-interference inner product. Queries and keys are projected into complex amplitude-phase representations. The attention score is:

```
score(i,j) = (1/√D) · Σ_d  Q_amp[d] · K_amp[d] · cos(Q_phase[d] − K_phase[d])
```

This is mathematically identical to the phase-coherence formula `Re[φ · exp(−iθ_probe)]` that achieves 30/30 perfect content-addressable retrieval in a nonlinear wave-field memory experiment. We call the resulting mechanism Moiré Attention.

In controlled comparisons on WikiText-2 (16M parameters, identical architecture except attention): Moiré Attention achieves 1.6% lower loss at 3 epochs, growing to **2.9% lower loss at 5 epochs**. The advantage widens monotonically across all epochs. At 138M parameters, 20 epochs on a mixed dataset (Guanaco + TinyStories + FineWeb), the trained model reaches avg loss **1.3702** and produces coherent conversational English.

---

## 1. The Formula

A 2D complex nonlinear wave field stores three phase-encoded memories. A probe wave with phase θ is injected. The retrieval score at memory location k is:

```
score_k = Σ Re[φ(r) · exp(-i · θ_probe)]
```

Matching memory: strong positive score. Non-matching: strong negative. 30/30 trials, zero false positives, ±0.1 radian noise tolerance. The discrimination ratio is approximately 3:1.

The same formula applied to token embeddings: project each token into `amp · exp(i·phase)`, compute the Hermitian inner product between query and key. Tokens whose phases align resonate (constructive interference, positive score). Tokens whose phases oppose cancel (destructive interference, negative score). This is Moiré Attention.

---

## 2. Architecture

```python
# Project to complex space
q_amp, q_phase = split(W_q · x)   # each [B, H, T, D]
k_amp, k_phase = split(W_k · x)
q_amp = softplus(q_amp)            # amplitudes positive
k_amp = softplus(k_amp)

# Phase-interference score (optimized: no 5D tensor)
q_real = q_amp * cos(q_phase)
q_imag = q_amp * sin(q_phase)
k_real = k_amp * cos(k_phase)
k_imag = k_amp * sin(k_phase)

scores = (matmul(q_real, k_real.T) + matmul(q_imag, k_imag.T)) * scale
```

This is Re[Q_c · conj(K_c)] where Q_c = Q_amp · exp(i·Q_phase). The cosine factor is bounded in [−1, 1] — natural regularization, no gradient explosion from extreme Q/K values.

**Theta-gamma gating.** Context is divided into gamma-rate slots. Each attention head has a learned theta offset θ_h that periodically modulates cross-slot attention:

```
gate(i,j) = cos(θ_h · (cycle_id(i) − cycle_id(j)))
scores    = scores * gate
```

Different heads learn different periodicities. The trained model shows theta offset diversity from −0.54 to +0.87 — the multiplexing is utilized, not ignored.

---

## 3. Results

### 16M parameter benchmark (WikiText-2)

| Epoch | Moiré loss | Standard loss | Gap |
|-------|-----------|--------------|-----|
| 1 | 5.805 | 5.861 | −0.056 |
| 2 | 4.735 | 4.818 | −0.083 |
| 3 | 4.272 | 4.368 | −0.096 |
| 4 | 3.992 | 4.100 | −0.109 |
| 5 | 3.855 | 3.970 | **−0.115 (2.9%)** |

The gap grows every epoch. This is not an initialization artifact.

### 138M parameter full training

Dataset: Guanaco + TinyStories + FineWeb, ~45M characters, ~49,000 sequences.  
Training: 20 epochs, ~493,000 steps, ~18.6 hours (RTX 3060).  
Final avg loss: **1.3702**

---

## 4. What the Cosine Factor Does

Standard attention: `Q · K`. Unbounded. Amplitude and direction conflated.

Moiré attention: `Q_amp · K_amp · cos(ΔΦ)`. The amplitude carries "how strongly to attend." The phase carries "what to attend to." The cosine factor is bounded — extreme values cannot dominate regardless of amplitude.

Two tokens with identical phase but different amplitudes: both attend to each other strongly (same content, different salience). Two tokens with opposing phase: negative score, destructive interference. Standard attention cannot represent the second case — it has no notion of phase opposition.

---

## 5. Connection to Wave Memory

Three independent results, one formula:

| Experiment | Formula | Result |
|-----------|---------|--------|
| Wave field retrieval | Re[φ · exp(−iθ)] | 30/30, zero false positives |
| EEG schizophrenia | Cross-band phase coherence | p=0.007, d=−1.21 |
| Language model | Re[Q_c · conj(K_c)] | 2.9% advantage, widening |

The formula was not chosen for elegance. It fell out of the wave memory experiment, was then recognized in the EEG result, then applied to attention. The direction of discovery matters: wave physics first, language model second.

---

## 6. Limitations

- Benchmark is 16M parameters on WikiText-2. Scaling behavior unknown.
- Parameter count is ~4% higher (2× Q/K projection width). Not equalized.
- No ablation of theta gating vs. phase scoring independently.
- Two runs, not a full statistical test.
- No comparison with Flash Attention, GQA, or other optimized variants.
- The cosine computation adds memory overhead — a fused kernel would be needed at scale.

---

## 7. Honest Assessment

**Demonstrated:** Phase-interference attention converges. It outperforms standard dot-product by 2.9% at 5 epochs, and the advantage widens. At 138M parameters it produces coherent English text.

**Not demonstrated:** Whether this advantage persists at scale. Whether it is the phase scoring or the theta gating or both. Whether the connection to biological phase coding is more than structural.

**Conjecture:** That the advantage reflects the richer angular similarity metric — the ability to represent phase opposition — rather than merely additional parameters. This would predict that the advantage grows with scale as the model learns more intricate phase structure.

---

## Code

`moire_conv_trainer_v5.py` — complete trainer with all dataset loaders.  
`moire_chat3.py` — interactive chat interface.

```bash
python moire_conv_trainer_v5.py --size xlarge --epochs 20 --batch_size 2
python moire_chat3.py --weights moire_phase2_weights_final.pt --size xlarge
```

---

*Antti Luode / PerceptionLab, Finland. Claude (Anthropic). March 2026.*
