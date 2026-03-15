"""
Deerskin Topological Explorer
==============================
Antti Luode (PerceptionLab, Finland) | 2026

A four-panel EEG visualizer through the Deerskin Architecture lens.
Neural computation as oscillatory phase-space geometry.

Panels:
  A — Macroscopic Moiré Field    (Betti-1 brain topography)
  B — Dendritic Delay Manifold   (3D Takens phase-space attractor)
  C — Theta Phase Gate           (PLV / gate rigidity over time)
  D — Cross-Band Eigenmode       (inter-band coupling network)

Dependencies:
    pip install PyQt5 matplotlib mne ripser persim scipy numpy
"""

import sys
import warnings
import threading
import traceback
import numpy as np

warnings.filterwarnings("ignore")

# ── Qt imports ────────────────────────────────────────────────────────────────
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QFileDialog, QProgressBar,
    QTextEdit, QGroupBox, QSlider, QComboBox, QSplitter, QFrame,
    QSizePolicy, QStatusBar, QTabWidget, QCheckBox, QSpinBox,
    QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon

# ── Matplotlib embedding ───────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ── Scientific stack ───────────────────────────────────────────────────────────
import mne
mne.set_log_level("WARNING")
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import ttest_ind
from itertools import combinations

try:
    from ripser import ripser
    RIPSER_OK = True
except ImportError:
    RIPSER_OK = False
    print("WARNING: ripser not found. Betti-1 will be estimated via SVD fallback.")

# ── ──────────────────────────────────────────────────────────────────────────
#   COLOUR PALETTE  (dark scientific theme)
# ── ──────────────────────────────────────────────────────────────────────────
BG_DARK   = "#0d0f14"
BG_PANEL  = "#12151c"
BG_CARD   = "#1a1e29"
ACCENT1   = "#00c8ff"   # cyan  – Betti/topology
ACCENT2   = "#ff6b35"   # orange – theta gate
ACCENT3   = "#a78bfa"   # violet – phase space
ACCENT4   = "#34d399"   # mint   – eigenmode
TEXT_HI   = "#e8eaf0"
TEXT_MED  = "#8892a4"
TEXT_DIM  = "#4a5568"
BORDER    = "#252b3b"

REGION_COLORS = {
    "Frontal":   "#00c8ff",
    "Temporal":  "#ff6b35",
    "Parietal":  "#a78bfa",
    "Occipital": "#34d399",
}

BAND_COLORS = {
    "delta": "#6366f1",
    "theta": "#f59e0b",
    "alpha": "#10b981",
    "beta":  "#ef4444",
    "gamma": "#ec4899",
}

# ── ──────────────────────────────────────────────────────────────────────────
#   EEG REGION DEFINITIONS
# ── ──────────────────────────────────────────────────────────────────────────
EEG_REGIONS = {
    "Frontal":   ["FP1","FP2","F3","F4","FZ","F7","F8","AF3","AF4"],
    "Temporal":  ["T3","T4","T7","T8","FT7","FT8","TP7","TP8"],
    "Parietal":  ["P3","P4","PZ","P7","P8","CP1","CP2","CP5","CP6"],
    "Occipital": ["O1","O2","OZ","PO3","PO4","PO7","PO8"],
}

BANDS = {
    "delta": (1,   4),
    "theta": (4,   8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

# ── ──────────────────────────────────────────────────────────────────────────
#   MATH / SIGNAL UTILITIES
# ── ──────────────────────────────────────────────────────────────────────────

def bandpass(signal: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    nyq = sfreq / 2.0
    low_n  = max(low  / nyq, 0.001)
    high_n = min(high / nyq, 0.999)
    b, a = butter(4, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal)


def takens_embed_3d(signal: np.ndarray, tau: int = 20) -> np.ndarray:
    """Embed 1-D signal into 3-D phase space with delay τ samples."""
    n = len(signal) - 2 * tau
    if n < 20:
        return None
    X = np.column_stack([
        signal[2*tau:],
        signal[tau : tau + n],
        signal[:n],
    ]).astype(np.float32)
    std = X.std(axis=0) + 1e-10
    X = (X - X.mean(axis=0)) / std
    return X


def compute_betti1(signal: np.ndarray, sfreq: float,
                   delays_ms=(10, 20, 40), subsample: int = 600,
                   persistence_threshold: float = 0.12) -> float:
    """Return Betti-1 persistence score.  Falls back to SVD rank if ripser absent."""
    if not RIPSER_OK:
        # SVD-based proxy: count eigenvalues above 5% of max
        X = takens_embed_3d(signal, tau=int(20 * sfreq / 1000))
        if X is None:
            return 0.0
        if len(X) > subsample:
            X = X[np.linspace(0, len(X)-1, subsample, dtype=int)]
        _, sv, _ = np.linalg.svd(X, full_matrices=False)
        sv = sv / (sv.max() + 1e-10)
        return float(np.sum(sv > 0.05))

    scores = []
    for d_ms in delays_ms:
        tau = max(1, int(d_ms * sfreq / 1000))
        X = takens_embed_3d(signal, tau=tau)
        if X is None:
            continue
        if len(X) > subsample:
            idx = np.linspace(0, len(X)-1, subsample, dtype=int)
            X = X[idx]
        try:
            result = ripser(X, maxdim=1)
            dgm = result["dgms"][1]
            if len(dgm) == 0:
                scores.append(0.0)
                continue
            lifetimes = dgm[:, 1] - dgm[:, 0]
            lifetimes = lifetimes[np.isfinite(lifetimes)]
            if len(lifetimes) == 0:
                scores.append(0.0)
                continue
            threshold = persistence_threshold * lifetimes.max()
            scores.append(float(lifetimes[lifetimes > threshold].sum()))
        except Exception:
            scores.append(0.0)
    return float(np.mean(scores)) if scores else 0.0


def compute_theta_plv_timeseries(signal: np.ndarray, sfreq: float,
                                  window_s: float = 2.0,
                                  step_s: float = 0.5) -> np.ndarray:
    """Sliding-window instantaneous theta PLV (single channel → phase consistency)."""
    theta = bandpass(signal, sfreq, 4, 8)
    phase = np.angle(hilbert(theta))
    win   = int(window_s * sfreq)
    step  = int(step_s  * sfreq)
    n     = len(phase)
    plvs  = []
    for start in range(0, n - win, step):
        seg = phase[start : start + win]
        plv = float(np.abs(np.mean(np.exp(1j * seg))))
        plvs.append(plv)
    return np.array(plvs)


def get_region_signal(data: np.ndarray, ch_names: list,
                      region: str, sfreq: float,
                      max_s: float = 60.0) -> np.ndarray:
    """Return mean-averaged signal for a brain region (robust to missing channels)."""
    wanted = [c.upper() for c in EEG_REGIONS.get(region, [])]
    ch_upper = [c.upper() for c in ch_names]
    idx = [i for i, c in enumerate(ch_upper) if c in wanted]
    if not idx:
        # fallback: positional heuristic
        n = len(ch_names)
        fallback = {
            "Frontal":   list(range(0, min(4, n))),
            "Temporal":  list(range(min(4, n), min(8, n))),
            "Parietal":  list(range(min(8, n), min(13, n))),
            "Occipital": list(range(min(13, n), min(17, n))),
        }
        idx = fallback.get(region, [0])
    n_samples = int(max_s * sfreq)
    seg = data[idx, :n_samples]
    return seg.mean(axis=0)


def compute_cross_band_coupling(data: np.ndarray, sfreq: float,
                                 max_s: float = 60.0) -> np.ndarray:
    """Return 5×5 correlation matrix of dominant eigenmode indices per band."""
    n_samples = int(max_s * sfreq)
    seg = data[:, :n_samples]
    band_names = list(BANDS.keys())
    sequences = []
    for band, (lo, hi) in BANDS.items():
        try:
            filt = np.array([bandpass(seg[c], sfreq, lo, hi)
                             for c in range(seg.shape[0])])
            power = (filt ** 2).mean(axis=1)  # mean power per channel
            sequences.append(power)
        except Exception:
            sequences.append(np.zeros(seg.shape[0]))

    # Correlation matrix across bands (channels as observations)
    mat = np.array(sequences)   # shape (5, n_channels)
    cov = np.corrcoef(mat)
    return cov, band_names


# ── ──────────────────────────────────────────────────────────────────────────
#   WORKER THREAD  (preprocessing + all heavy computation)
# ── ──────────────────────────────────────────────────────────────────────────

class AnalysisPanel(QWidget):
    """Batch statistics panel — shown as a tab in the log area."""

    # ── Data-quality thresholds ───────────────────────────────────────────
    COUPLING_CEILING = 0.95   # cross-band coupling > this → hardware artifact
    BETTI_FLOOR      = 5.0    # min regional Betti-1 < this → destroyed topology

    def __init__(self, parent=None):
        super().__init__(parent)
        self.log_fn = None          # optional callback for logging
        self.excluded = []          # list of (filename, reason) tuples
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        # 4 subplot columns: plot1, plot2, heatmap, heatmap-colorbar
        self.fig = plt.figure(figsize=(9, 2.8), facecolor=BG_DARK)
        gs = self.fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.45)
        self.axes = [self.fig.add_subplot(gs[i]) for i in range(3)]
        self.ax_hm_cb = self.fig.add_subplot(gs[3])
        self.ax_hm_cb.set_visible(False)
        for ax in self.axes:
            ax.set_facecolor(BG_PANEL)
        self.fig.tight_layout(pad=1.5)
        self.canvas = _styled_canvas(self.fig)
        lay.addWidget(self.canvas)

    def _log(self, msg: str):
        if self.log_fn:
            self.log_fn(msg)

    def _is_artifact(self, fname: str, r: dict) -> str:
        """Return exclusion reason or empty string if clean."""
        mat = r.get("coupling_matrix")
        bnames = r.get("band_names", list(BANDS.keys()))
        # Compute mean coupling from numpy matrix
        if mat is not None and not isinstance(mat, dict):
            n = mat.shape[0]
            off = [abs(mat[i, j]) for i in range(n) for j in range(n) if i != j]
            mean_c = float(np.mean(off)) if off else 0.0
        elif isinstance(mat, dict):
            off = [abs(mat[b1][b2]) for b1 in bnames for b2 in bnames if b1 != b2]
            mean_c = float(np.mean(off)) if off else 0.0
        else:
            mean_c = 0.0
        if mean_c > self.COUPLING_CEILING:
            return f"coupling={mean_c:.3f} > {self.COUPLING_CEILING} (hardware artifact)"

        betti = r.get("betti_scores", {})
        if betti:
            min_b = min(betti.values())
            if min_b < self.BETTI_FLOOR:
                return f"min Betti-1={min_b:.2f} < {self.BETTI_FLOOR} (destroyed topology)"
        return ""

    def run(self, batch_results: list):
        """batch_results: list of (filename, results_dict)"""
        from scipy.stats import ttest_ind

        # ── Artifact exclusion ────────────────────────────────────────────
        clean = []
        self.excluded = []
        for f, r in batch_results:
            if 'error' in r:
                continue
            reason = self._is_artifact(f, r)
            if reason:
                self.excluded.append((f, reason))
                self._log(f"  ⚠ EXCLUDED {f}: {reason}")
            else:
                clean.append((f, r))

        if self.excluded:
            self._log(f"  Data quality filter: {len(self.excluded)} file(s) excluded, "
                       f"{len(clean)} retained")
        else:
            self._log(f"  Data quality filter: all {len(clean)} files passed")

        hc = [(f, r) for f, r in clean if f.lower().startswith('h')]
        sz = [(f, r) for f, r in clean if f.lower().startswith('s')]
        if not hc or not sz:
            self._log("  Not enough HC/SZ files after exclusion for group analysis.")
            return

        self._log(f"  Group analysis: {len(hc)} HC vs {len(sz)} SZ")

        regions = list(EEG_REGIONS.keys())
        bands   = list(BANDS.keys())

        # ── Derive per-result metrics ─────────────────────────────────────
        for _, r in hc + sz:
            mat = r.get("coupling_matrix")
            bnames = r.get("band_names", bands)
            if mat is not None and not isinstance(mat, dict):
                n = mat.shape[0]
                off_diag = [abs(mat[i, j]) for i in range(n)
                            for j in range(n) if i != j]
                r["mean_cross_band_coupling"] = float(
                    np.mean(off_diag)) if off_diag else 0.0
                r["coupling_matrix_dict"] = {
                    bnames[i]: {bnames[j]: float(mat[i, j])
                                for j in range(n)}
                    for i in range(n)
                }
            elif isinstance(mat, dict):
                if "mean_cross_band_coupling" not in r:
                    off = [abs(mat[b1][b2]) for b1 in bnames for b2 in bnames if b1 != b2]
                    r["mean_cross_band_coupling"] = float(np.mean(off)) if off else 0.0
                if "coupling_matrix_dict" not in r:
                    r["coupling_matrix_dict"] = mat
            elif mat is None:
                r["mean_cross_band_coupling"] = 0.0
                r["coupling_matrix_dict"] = {b1: {b2: 0.0 for b2 in bands}
                                              for b1 in bands}

        # ── Clear axes ────────────────────────────────────────────────────
        for ax in self.axes:
            ax.clear()
            ax.set_facecolor(BG_PANEL)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)
            ax.tick_params(colors=TEXT_DIM, labelsize=6)
        self.ax_hm_cb.clear()
        self.ax_hm_cb.set_visible(False)

        # ── Plot 1: Betti-1 per region ───────────────────────────────────
        ax = self.axes[0]
        x = np.arange(len(regions))
        w = 0.35
        hc_b = [[r['betti_scores'][reg] for _, r in hc] for reg in regions]
        sz_b = [[r['betti_scores'][reg] for _, r in sz] for reg in regions]
        ax.bar(x - w/2, [np.mean(v) for v in hc_b], w,
               color=ACCENT4, alpha=0.8, label="HC")
        ax.bar(x + w/2, [np.mean(v) for v in sz_b], w,
               color=ACCENT2, alpha=0.8, label="SZ")
        ax.errorbar(x - w/2, [np.mean(v) for v in hc_b],
                    yerr=[np.std(v) for v in hc_b],
                    fmt='none', color='white', capsize=2, lw=0.8)
        ax.errorbar(x + w/2, [np.mean(v) for v in sz_b],
                    yerr=[np.std(v) for v in sz_b],
                    fmt='none', color='white', capsize=2, lw=0.8)
        # significance stars
        for i, (hv, sv) in enumerate(zip(hc_b, sz_b)):
            _, p = ttest_ind(hv, sv)
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if star:
                ymax = max(np.mean(hv) + np.std(hv), np.mean(sv) + np.std(sv))
                ax.text(i, ymax + 0.3, star, ha='center', fontsize=7,
                        color='white', fontfamily='Courier New')
        ax.set_xticks(x)
        ax.set_xticklabels([r[:4] for r in regions], fontsize=6, color=TEXT_MED)
        ax.set_ylabel("Betti-1", fontsize=7, color=TEXT_MED)
        ax.set_title("Betti-1 by Region", fontsize=7, color=ACCENT1,
                     fontfamily="Courier New")
        ax.legend(fontsize=6, labelcolor=TEXT_MED,
                  facecolor=BG_CARD, edgecolor=BORDER)
        ax.set_facecolor(BG_PANEL)

        # ── Plot 2: Mean cross-band coupling HC vs SZ ─────────────────────
        ax = self.axes[1]
        hc_c = [r['mean_cross_band_coupling'] for _, r in hc]
        sz_c = [r['mean_cross_band_coupling'] for _, r in sz]
        _, p_c = ttest_ind(hc_c, sz_c)

        parts = ax.violinplot([hc_c, sz_c], positions=[0, 1],
                               showmeans=True, showmedians=False)
        for i, (pc, col) in enumerate(zip(parts['bodies'], [ACCENT4, ACCENT2])):
            pc.set_facecolor(col)
            pc.set_alpha(0.6)
        parts['cmeans'].set_color('white')
        parts['cmeans'].set_linewidth(1.5)
        for key in ('cbars', 'cmins', 'cmaxes'):
            parts[key].set_color(BORDER)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['HC', 'SZ'], fontsize=7, color=TEXT_MED)
        ax.set_ylabel("|coupling|", fontsize=7, color=TEXT_MED)
        ax.set_title(f"Cross-Band Coupling  p={p_c:.3f}", fontsize=7,
                     color=ACCENT4 if p_c < 0.05 else TEXT_MED,
                     fontfamily="Courier New")
        ax.set_facecolor(BG_PANEL)
        if p_c < 0.05:
            ymax = max(max(hc_c), max(sz_c))
            ax.annotate("", xy=(1, ymax*1.02), xytext=(0, ymax*1.02),
                        arrowprops=dict(arrowstyle="-", color="white", lw=0.8))
            star = "***" if p_c<0.001 else "**" if p_c<0.01 else "*"
            ax.text(0.5, ymax*1.04, star, ha='center', fontsize=8,
                    color='white', fontfamily='Courier New')

        # ── Plot 3: Per-band coupling HC vs SZ heatmap ───────────────────
        ax = self.axes[2]
        hc_mat = np.zeros((len(bands), len(bands)))
        sz_mat = np.zeros((len(bands), len(bands)))
        p_mat  = np.ones((len(bands), len(bands)))
        for i, b1 in enumerate(bands):
            for j, b2 in enumerate(bands):
                if i == j:
                    hc_mat[i,j] = sz_mat[i,j] = 1.0
                    continue
                hc_v = [abs(r['coupling_matrix_dict'][b1][b2]) for _, r in hc]
                sz_v = [abs(r['coupling_matrix_dict'][b1][b2]) for _, r in sz]
                hc_mat[i,j] = np.mean(hc_v)
                sz_mat[i,j] = np.mean(sz_v)
                _, p_mat[i,j] = ttest_ind(hc_v, sz_v)

        diff_mat = sz_mat - hc_mat
        im = ax.imshow(diff_mat, cmap='RdBu_r', vmin=-0.4, vmax=0.4, aspect='auto')
        ax.set_xticks(range(len(bands)))
        ax.set_yticks(range(len(bands)))
        ax.set_xticklabels([b[:3] for b in bands], rotation=45, fontsize=5.5,
                            color=TEXT_MED)
        ax.set_yticklabels([b[:3] for b in bands], fontsize=5.5, color=TEXT_MED)
        for i in range(len(bands)):
            for j in range(len(bands)):
                if i != j:
                    p = p_mat[i,j]
                    marker = "✦" if p < 0.05 else ""
                    val = diff_mat[i,j]
                    ax.text(j, i, f"{val:+.2f}{marker}", ha='center', va='center',
                            fontsize=4.5, color='white' if abs(val)>0.2 else TEXT_DIM)
        cbar2 = self.fig.colorbar(im, cax=self.ax_hm_cb)
        self.ax_hm_cb.set_visible(True)
        cbar2.ax.tick_params(colors=TEXT_DIM, labelsize=5)
        cbar2.outline.set_edgecolor(BORDER)
        ax.set_title("SZ−HC coupling diff  (✦ p<.05)", fontsize=6.5,
                     color=ACCENT2, fontfamily="Courier New")
        ax.set_facecolor(BG_PANEL)

        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()

class AnalysisWorker(QThread):

    progress   = pyqtSignal(int, str)
    finished   = pyqtSignal(dict)
    error      = pyqtSignal(str)

    def __init__(self, filepath: str, ica_enabled: bool = True,
                 analysis_duration: float = 60.0):
        super().__init__()
        self.filepath = filepath
        self.ica_enabled = ica_enabled
        self.analysis_duration = analysis_duration

    def run(self):
        try:
            self._run()
        except Exception as e:
            self.error.emit(f"Analysis failed:\n{traceback.format_exc()}")

    def _run(self):
        results = {}
        self.progress.emit(5, "Loading EDF …")

        raw = mne.io.read_raw_edf(
            self.filepath, preload=True, verbose=False
        )
        # Normalise channel names
        raw.rename_channels(lambda n: n.strip().upper()
                                        .replace(" ", "")
                                        .replace(".", "")
                                        .replace("-", ""))
        sfreq = raw.info["sfreq"]
        ch_names = raw.ch_names
        results["sfreq"]    = sfreq
        results["ch_names"] = ch_names

        self.progress.emit(15, "Bandpass filtering 1–45 Hz …")
        raw.filter(1.0, 45.0, fir_design="firwin", verbose=False)

        # ── ICA artifact removal ──────────────────────────────────────────
        if self.ica_enabled:
            self.progress.emit(25, "Running FastICA artifact rejection …")
            try:
                n_comp = min(15, len(ch_names) - 1)
                ica = mne.preprocessing.ICA(
                    n_components=n_comp, method="fastica", random_state=42
                )
                ica.fit(raw, verbose=False)

                # Find EOG-like components: correlate with FP1/FP2
                fp_candidates = [c for c in ch_names
                                 if "FP1" in c or "FP2" in c or "FPZ" in c]
                if fp_candidates:
                    eog_idx, _ = ica.find_bads_eog(
                        raw, ch_name=fp_candidates[0], verbose=False
                    )
                    ica.exclude = eog_idx[:2]
                    ica.apply(raw, verbose=False)
                    results["ica_removed"] = eog_idx[:2]
                else:
                    results["ica_removed"] = []
            except Exception as e:
                results["ica_warning"] = str(e)

        data = raw.get_data()
        results["data"] = data

        # ── Region Betti-1 scores ─────────────────────────────────────────
        self.progress.emit(40, "Computing Betti-1 topological complexity …")
        betti_scores = {}
        for region in EEG_REGIONS:
            sig = get_region_signal(data, ch_names, region, sfreq,
                                     self.analysis_duration)
            betti_scores[region] = compute_betti1(sig, sfreq)
        results["betti_scores"] = betti_scores

        # ── Per-region Takens embeddings (3-D, subsampled for plotting) ───
        self.progress.emit(55, "Building phase-space attractor clouds …")
        attractors = {}
        for region in EEG_REGIONS:
            sig = get_region_signal(data, ch_names, region, sfreq,
                                     self.analysis_duration)
            tau = max(1, int(20 * sfreq / 1000))
            X = takens_embed_3d(sig, tau=tau)
            if X is not None and len(X) > 800:
                idx = np.linspace(0, len(X)-1, 800, dtype=int)
                X = X[idx]
            attractors[region] = X
        results["attractors"] = attractors

        # ── Theta PLV timeseries per region ───────────────────────────────
        self.progress.emit(70, "Computing theta gate rigidity …")
        plv_series = {}
        for region in EEG_REGIONS:
            sig = get_region_signal(data, ch_names, region, sfreq,
                                     self.analysis_duration)
            plv_series[region] = compute_theta_plv_timeseries(sig, sfreq)
        results["plv_series"] = plv_series

        # ── Cross-band coupling matrix ────────────────────────────────────
        self.progress.emit(85, "Computing cross-band eigenmode coupling …")
        cov, band_names = compute_cross_band_coupling(
            data, sfreq, self.analysis_duration
        )
        results["coupling_matrix"] = cov
        results["band_names"] = band_names

        self.progress.emit(100, "Analysis complete.")
        self.finished.emit(results)


# ── ──────────────────────────────────────────────────────────────────────────
#   PANEL WIDGETS
# ── ──────────────────────────────────────────────────────────────────────────

def _styled_canvas(fig, parent=None):
    canvas = FigureCanvas(fig)
    canvas.setStyleSheet(f"background:{BG_PANEL}; border:1px solid {BORDER};")
    return canvas


class PanelA(QWidget):
    """Macroscopic Moiré Field — Betti-1 regional brain map."""

    region_clicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.betti_scores = {}

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        hdr = QLabel("A — Macroscopic Moiré Field")
        hdr.setStyleSheet(f"color:{ACCENT1}; font-size:11px; font-weight:bold;"
                          f"font-family:'Courier New'; padding:3px 0;")
        lay.addWidget(hdr)

        # Two axes: brain map + fixed colorbar column (no resizing on redraw)
        self.fig = plt.figure(figsize=(4, 3.5), facecolor=BG_PANEL)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 0.07], wspace=0.05)
        self.ax   = self.fig.add_subplot(gs[0])
        self.ax_cb = self.fig.add_subplot(gs[1])
        self.ax.set_facecolor(BG_PANEL)
        self.ax_cb.set_facecolor(BG_PANEL)
        self._cbar = None
        self.canvas = _styled_canvas(self.fig)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        lay.addWidget(self.canvas)

        sub = QLabel("Click a region to inspect")
        sub.setStyleSheet(f"color:{TEXT_DIM}; font-size:9px; font-family:'Courier New';")
        sub.setAlignment(Qt.AlignCenter)
        lay.addWidget(sub)

    # Simple 2-D brain schematic with 4+1 regions as coloured patches
    # Left temporal at 0.18, Right temporal at 0.82 (mirror)
    _REGION_XY = {
        "Frontal":          (0.50, 0.82),
        "Parietal":         (0.50, 0.55),
        "Temporal (L)":     (0.18, 0.55),
        "Temporal (R)":     (0.82, 0.55),
        "Occipital":        (0.50, 0.24),
    }
    _REGION_ELLIPSE = {
        "Frontal":          (0.28, 0.20),
        "Parietal":         (0.28, 0.24),
        "Temporal (L)":     (0.18, 0.22),
        "Temporal (R)":     (0.18, 0.22),
        "Occipital":        (0.22, 0.20),
    }
    # Map display names to real region keys
    _DISPLAY_TO_REGION = {
        "Frontal":      "Frontal",
        "Parietal":     "Parietal",
        "Temporal (L)": "Temporal",
        "Temporal (R)": "Temporal",
        "Occipital":    "Occipital",
    }

    def update_scores(self, betti_scores: dict):
        self.betti_scores = betti_scores
        self._draw()

    def _draw(self):
        self.ax.clear()
        self.ax.set_facecolor(BG_PANEL)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1.05)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        # Draw scalp outline
        theta = np.linspace(0, 2 * np.pi, 200)
        cx, cy, r = 0.5, 0.53, 0.40
        self.ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
                     color=BORDER, lw=1.5, zorder=1)
        # Nose
        nose_t = np.linspace(-0.2, 0.2, 30)
        self.ax.plot(cx + nose_t, cy + r + 0.02 - 3 * nose_t**2,
                     color=BORDER, lw=1, zorder=1)
        # Ears
        for ex in [0.10, 0.90]:
            self.ax.plot([ex, ex], [0.46, 0.60], color=BORDER, lw=1.5, zorder=1)

        display_colors = {
            "Frontal":      REGION_COLORS["Frontal"],
            "Parietal":     REGION_COLORS["Parietal"],
            "Temporal (L)": REGION_COLORS["Temporal"],
            "Temporal (R)": REGION_COLORS["Temporal"],
            "Occipital":    REGION_COLORS["Occipital"],
        }

        if not self.betti_scores:
            for dname, (rx, ry) in self._REGION_XY.items():
                ew, eh = self._REGION_ELLIPSE[dname]
                patch = matplotlib.patches.Ellipse(
                    (rx, ry), ew, eh,
                    color=display_colors[dname], alpha=0.18, zorder=2
                )
                self.ax.add_patch(patch)
                label = dname.replace(" (L)","").replace(" (R)","")
                self.ax.text(rx, ry, label, ha="center", va="center",
                             fontsize=6, color=display_colors[dname],
                             fontfamily="Courier New", zorder=3)
            self.ax_cb.set_visible(False)
            self.canvas.draw()
            return

        # Build display scores: Temporal (L) and (R) share the same Temporal value
        display_scores = {}
        for dname, rname in self._DISPLAY_TO_REGION.items():
            if rname in self.betti_scores:
                display_scores[dname] = self.betti_scores[rname]

        vals = np.array(list(display_scores.values()), dtype=float)
        vmin, vmax = vals.min(), vals.max()
        if vmax == vmin:
            vmax = vmin + 1

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap("plasma")

        for dname, score in display_scores.items():
            rx, ry = self._REGION_XY[dname]
            ew, eh = self._REGION_ELLIPSE[dname]
            base_color = np.array(mcolors.to_rgba(cmap(norm(score))))
            patch = matplotlib.patches.Ellipse(
                (rx, ry), ew, eh,
                facecolor=base_color, edgecolor=display_colors[dname],
                lw=1.5, alpha=0.85, zorder=2
            )
            self.ax.add_patch(patch)
            label = dname.replace(" (L)","").replace(" (R)","")
            side  = " L" if "(L)" in dname else (" R" if "(R)" in dname else "")
            self.ax.text(rx, ry + 0.01, label + side, ha="center", va="center",
                         fontsize=6, color="white",
                         fontfamily="Courier New", fontweight="bold", zorder=4)
            self.ax.text(rx, ry - 0.07, f"β₁={score:.1f}", ha="center",
                         fontsize=5.5, color=display_colors[dname],
                         fontfamily="Courier New", zorder=4)

        # Colorbar in dedicated axis — clear and redraw (no fig-level accumulation)
        self.ax_cb.clear()
        self.ax_cb.set_facecolor(BG_PANEL)
        self.ax_cb.set_visible(True)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        self._cbar = self.fig.colorbar(sm, cax=self.ax_cb, label="Betti-1")
        self._cbar.ax.yaxis.label.set_color(TEXT_MED)
        self._cbar.ax.tick_params(colors=TEXT_MED, labelsize=5)
        self._cbar.outline.set_edgecolor(BORDER)

        self.ax.set_title("Topological Complexity Map", fontsize=7,
                           color=TEXT_MED, fontfamily="Courier New", pad=4)
        self.canvas.draw()

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        for dname, (rx, ry) in self._REGION_XY.items():
            ew, eh = self._REGION_ELLIPSE[dname]
            dx = (event.xdata - rx) / (ew / 2)
            dy = (event.ydata - ry) / (eh / 2)
            if dx*dx + dy*dy <= 1.0:
                real_region = self._DISPLAY_TO_REGION[dname]
                self.region_clicked.emit(real_region)
                return


class PanelB(QWidget):
    """Dendritic Delay Manifold — rotating 3-D Takens attractor."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._attractor = None
        self._angle = 0
        self._setup_ui()

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        hdr = QLabel("B — Dendritic Delay Manifold")
        hdr.setStyleSheet(f"color:{ACCENT3}; font-size:11px; font-weight:bold;"
                          f"font-family:'Courier New'; padding:3px 0;")
        lay.addWidget(hdr)

        self.fig = plt.figure(figsize=(4, 3.5), facecolor=BG_PANEL)
        self.ax3d = self.fig.add_subplot(111, projection="3d")
        self._style_3d()
        self.canvas = _styled_canvas(self.fig)
        lay.addWidget(self.canvas)

        self._region_label = QLabel("Select a region in Panel A")
        self._region_label.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:9px; font-family:'Courier New';"
        )
        self._region_label.setAlignment(Qt.AlignCenter)
        lay.addWidget(self._region_label)

        # Auto-rotation timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._rotate)
        self._timer.start(80)

    def _style_3d(self):
        ax = self.ax3d
        ax.set_facecolor(BG_PANEL)
        self.fig.patch.set_facecolor(BG_PANEL)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor(BORDER)
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.line.set_color(BORDER)
            axis.set_tick_params(colors=TEXT_DIM, labelsize=5)

    def set_attractor(self, X: np.ndarray, region: str):
        self._attractor = X
        self._region = region
        self._draw_static()

    def _draw_static(self):
        self.ax3d.clear()
        self._style_3d()
        if self._attractor is None or len(self._attractor) == 0:
            self.canvas.draw()
            return
        X = self._attractor
        colors = cm.plasma(np.linspace(0, 1, len(X)))
        self.ax3d.scatter(X[:, 0], X[:, 1], X[:, 2],
                          c=colors, s=0.8, alpha=0.6, depthshade=True)
        self.ax3d.set_title(f"Phase-Space: {self._region}",
                             fontsize=8, color=ACCENT3,
                             fontfamily="Courier New", pad=2)
        self.ax3d.set_xlabel("x(t)", fontsize=5, color=TEXT_DIM)
        self.ax3d.set_ylabel("x(t-τ)", fontsize=5, color=TEXT_DIM)
        self.ax3d.set_zlabel("x(t-2τ)", fontsize=5, color=TEXT_DIM)
        self.canvas.draw()

    def _rotate(self):
        if self._attractor is None:
            return
        self._angle = (self._angle + 1.5) % 360
        self.ax3d.view_init(elev=20, azim=self._angle)
        self.canvas.draw_idle()


class PanelC(QWidget):
    """Theta Phase Gate — PLV timeseries with gate state indicator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        hdr = QLabel("C — Theta Phase Gate")
        hdr.setStyleSheet(f"color:{ACCENT2}; font-size:11px; font-weight:bold;"
                          f"font-family:'Courier New'; padding:3px 0;")
        lay.addWidget(hdr)

        self.fig, (self.ax_plv, self.ax_gauge) = plt.subplots(
            1, 2, figsize=(5, 3.0), facecolor=BG_PANEL,
            gridspec_kw={"width_ratios": [3, 1]}
        )
        for ax in (self.ax_plv, self.ax_gauge):
            ax.set_facecolor(BG_PANEL)
        self.fig.tight_layout(pad=1.2)
        self.canvas = _styled_canvas(self.fig)
        lay.addWidget(self.canvas)

        # Gate state text
        self.gate_label = QLabel("Gate: —")
        self.gate_label.setStyleSheet(
            f"color:{TEXT_MED}; font-size:10px; font-weight:bold;"
            f"font-family:'Courier New'; padding:2px;"
        )
        self.gate_label.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.gate_label)

    def update_plv(self, plv_series: np.ndarray, region: str):
        self.ax_plv.clear()
        self.ax_gauge.clear()
        for ax in (self.ax_plv, self.ax_gauge):
            ax.set_facecolor(BG_PANEL)

        if plv_series is None or len(plv_series) == 0:
            self.canvas.draw()
            return

        t = np.arange(len(plv_series)) * 0.5  # 0.5 s step
        mean_plv = float(np.mean(plv_series))
        std_plv  = float(np.std(plv_series))

        # Classify gate state
        # Healthy: PLV ~ 0.4–0.65, moderate variance
        # Hijacked: PLV > 0.65, low variance → schizophrenic signature
        # Degraded: PLV < 0.3 → Alzheimer-like
        if mean_plv > 0.62 and std_plv < 0.07:
            gate_state = "HIJACKED"
            gate_color = ACCENT2
        elif mean_plv < 0.35:
            gate_state = "DEGRADED"
            gate_color = BAND_COLORS["delta"]
        else:
            gate_state = "HEALTHY"
            gate_color = ACCENT4

        # PLV timeseries plot
        self.ax_plv.fill_between(t, plv_series, alpha=0.25, color=ACCENT2)
        self.ax_plv.plot(t, plv_series, color=ACCENT2, lw=1.2)
        self.ax_plv.axhline(mean_plv, color=gate_color, lw=1.0, ls="--",
                             label=f"μ={mean_plv:.3f}")
        self.ax_plv.fill_between(t, mean_plv - std_plv, mean_plv + std_plv,
                                  color=gate_color, alpha=0.08)
        self.ax_plv.set_ylim(0, 1)
        self.ax_plv.set_xlabel("Time (s)", fontsize=7, color=TEXT_DIM)
        self.ax_plv.set_ylabel("Theta PLV", fontsize=7, color=TEXT_DIM)
        self.ax_plv.tick_params(colors=TEXT_DIM, labelsize=6)
        for sp in self.ax_plv.spines.values():
            sp.set_edgecolor(BORDER)
        self.ax_plv.legend(fontsize=6, labelcolor=TEXT_MED,
                            facecolor=BG_CARD, edgecolor=BORDER)
        self.ax_plv.set_title(f"Theta PLV: {region}", fontsize=7,
                               color=TEXT_MED, fontfamily="Courier New")

        # Gauge (semi-circular arc)
        gauge_t = np.linspace(np.pi, 0, 180)
        self.ax_gauge.plot(np.cos(gauge_t), np.sin(gauge_t),
                            color=BORDER, lw=6)
        filled_t = np.linspace(np.pi, np.pi - mean_plv * np.pi, 180)
        self.ax_gauge.plot(np.cos(filled_t), np.sin(filled_t),
                            color=gate_color, lw=6)
        self.ax_gauge.text(0, -0.25, f"{mean_plv:.2f}", ha="center",
                            fontsize=10, color=gate_color,
                            fontfamily="Courier New", fontweight="bold")
        self.ax_gauge.set_xlim(-1.2, 1.2)
        self.ax_gauge.set_ylim(-0.4, 1.1)
        self.ax_gauge.axis("off")

        self.gate_label.setText(f"Gate: {gate_state}")
        self.gate_label.setStyleSheet(
            f"color:{gate_color}; font-size:11px; font-weight:bold;"
            f"font-family:'Courier New'; padding:2px; "
            f"background:{BG_CARD}; border-radius:4px;"
        )

        self.canvas.draw()


class PanelD(QWidget):
    """Cross-Band Eigenmode Coupling — correlation heatmap + network graph."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        hdr = QLabel("D — Cross-Band Eigenmode Coupling")
        hdr.setStyleSheet(f"color:{ACCENT4}; font-size:11px; font-weight:bold;"
                          f"font-family:'Courier New'; padding:3px 0;")
        lay.addWidget(hdr)

        self.fig = plt.figure(figsize=(5, 3.2), facecolor=BG_PANEL)
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 0.06, 1], wspace=0.35)
        self.ax_mat = self.fig.add_subplot(gs[0])
        self.ax_cb  = self.fig.add_subplot(gs[1])
        self.ax_net = self.fig.add_subplot(gs[2])
        for ax in (self.ax_mat, self.ax_net):
            ax.set_facecolor(BG_PANEL)
        self.ax_cb.set_visible(False)
        self.fig.tight_layout(pad=1.2)
        self.canvas = _styled_canvas(self.fig)
        lay.addWidget(self.canvas)

        self.coupling_label = QLabel("Mean coupling: —")
        self.coupling_label.setStyleSheet(
            f"color:{TEXT_MED}; font-size:9px; font-family:'Courier New';"
        )
        self.coupling_label.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.coupling_label)

    def update_coupling(self, matrix, band_names: list):
        self.ax_mat.clear()
        self.ax_cb.clear()
        self.ax_net.clear()
        for ax in (self.ax_mat, self.ax_net):
            ax.set_facecolor(BG_PANEL)

        if matrix is None:
            self.ax_cb.set_visible(False)
            self.canvas.draw()
            return

        # Accept dict-of-dicts or numpy array
        if isinstance(matrix, dict):
            matrix = np.array([[matrix[b1][b2] for b2 in band_names]
                               for b1 in band_names], dtype=float)

        n = len(band_names)

        # Heatmap
        im = self.ax_mat.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1,
                                 aspect="auto")
        self.ax_mat.set_xticks(range(n))
        self.ax_mat.set_yticks(range(n))
        self.ax_mat.set_xticklabels(band_names, rotation=45, fontsize=6,
                                     color=TEXT_MED)
        self.ax_mat.set_yticklabels(band_names, fontsize=6, color=TEXT_MED)
        self.ax_mat.tick_params(colors=TEXT_DIM)
        for sp in self.ax_mat.spines.values():
            sp.set_edgecolor(BORDER)
        for i in range(n):
            for j in range(n):
                val = matrix[i, j]
                self.ax_mat.text(j, i, f"{val:.2f}", ha="center", va="center",
                                  fontsize=5.5,
                                  color="white" if abs(val) > 0.5 else TEXT_DIM)
        # Colorbar in dedicated axis — no accumulation
        self.ax_cb.set_visible(True)
        self.ax_cb.set_facecolor(BG_PANEL)
        cbar = self.fig.colorbar(im, cax=self.ax_cb)
        cbar.ax.tick_params(colors=TEXT_DIM, labelsize=5)
        cbar.outline.set_edgecolor(BORDER)
        self.ax_mat.set_title("Coupling matrix", fontsize=7,
                               color=TEXT_MED, fontfamily="Courier New")

        # Network graph (spring layout)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}

        band_color_list = [BAND_COLORS.get(b, ACCENT1) for b in band_names]

        # Draw edges above threshold
        threshold = 0.3
        for i, j in combinations(range(n), 2):
            w = abs(matrix[i, j])
            if w > threshold:
                x0, y0 = pos[i]
                x1, y1 = pos[j]
                lw = float(w * 4)
                alpha = float(min(1.0, w))
                color = ACCENT2 if w > 0.6 else ACCENT4
                self.ax_net.plot([x0, x1], [y0, y1],
                                  color=color, lw=lw, alpha=alpha * 0.7, zorder=1)

        # Draw nodes
        for i, bname in enumerate(band_names):
            x, y = pos[i]
            c = band_color_list[i]
            self.ax_net.scatter(x, y, s=140, color=c, zorder=3,
                                 edgecolors="white", linewidths=0.5)
            self.ax_net.text(x * 1.25, y * 1.25, bname, ha="center",
                              fontsize=6.5, color=c,
                              fontfamily="Courier New", fontweight="bold")

        self.ax_net.set_xlim(-1.6, 1.6)
        self.ax_net.set_ylim(-1.6, 1.6)
        self.ax_net.axis("off")
        self.ax_net.set_title("Band coupling network", fontsize=7,
                               color=TEXT_MED, fontfamily="Courier New")

        # Summary stat
        off_diag = [(i, j) for i in range(n) for j in range(n) if i != j]
        mean_c = np.mean([abs(matrix[i, j]) for i, j in off_diag])
        self.coupling_label.setText(f"Mean |coupling|: {mean_c:.3f}")

        self.canvas.draw()


# ── ──────────────────────────────────────────────────────────────────────────
#   LOG PANEL
# ── ──────────────────────────────────────────────────────────────────────────

class LogPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setStyleSheet(
            f"background:{BG_DARK}; color:{TEXT_MED}; "
            f"font-family:'Courier New'; font-size:9px; "
            f"border:1px solid {BORDER};"
        )
        lay.addWidget(self.text)

    def log(self, msg: str):
        self.text.append(f"▸ {msg}")
        self.text.ensureCursorVisible()

    def clear(self):
        self.text.clear()


# ── ──────────────────────────────────────────────────────────────────────────
#   FOLDER BATCH WORKER
# ── ──────────────────────────────────────────────────────────────────────────

class FolderWorker(QThread):
    """Sequentially analyses every EDF in a folder, emitting results one by one."""
    file_started   = pyqtSignal(int, int, str)      # (index, total, filename)
    file_done      = pyqtSignal(int, str, dict)      # (index, filename, results)
    file_error     = pyqtSignal(int, str, str)       # (index, filename, error)
    all_done       = pyqtSignal(list)                # list of (filename, results) tuples
    sub_progress   = pyqtSignal(int, str)            # forwarded from sub-workers

    def __init__(self, filepaths: list, ica_enabled: bool = True,
                 analysis_duration: float = 60.0):
        super().__init__()
        self.filepaths = filepaths
        self.ica_enabled = ica_enabled
        self.analysis_duration = analysis_duration
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        completed = []
        for i, fp in enumerate(self.filepaths):
            if self._abort:
                break
            fname = fp.replace("\\", "/").split("/")[-1]
            self.file_started.emit(i, len(self.filepaths), fname)
            try:
                worker = AnalysisWorker(fp, self.ica_enabled, self.analysis_duration)
                # Run synchronously inside this thread
                worker._run()
                # worker._run() stores results via self.finished – but we called
                # it directly, so we collect from internal state.  Instead we
                # re-run via a thin wrapper:
                results = _run_analysis_sync(fp, self.ica_enabled, self.analysis_duration,
                                              lambda p, m: self.sub_progress.emit(p, m))
                self.file_done.emit(i, fname, results)
                completed.append((fname, results))
            except Exception as e:
                self.file_error.emit(i, fname, traceback.format_exc())
                completed.append((fname, {"error": str(e)}))
        self.all_done.emit(completed)


def _run_analysis_sync(filepath, ica_enabled, analysis_duration, progress_cb):
    """Synchronous version of AnalysisWorker._run() for use inside FolderWorker."""
    results = {}
    progress_cb(5, f"Loading {filepath.split('/')[-1]} …")
    raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    raw.rename_channels(lambda n: n.strip().upper()
                                    .replace(" ", "").replace(".", "").replace("-", ""))
    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names
    results["sfreq"] = sfreq
    results["ch_names"] = ch_names

    progress_cb(15, "Bandpass …")
    raw.filter(1.0, 45.0, fir_design="firwin", verbose=False)

    if ica_enabled:
        progress_cb(25, "ICA …")
        try:
            n_comp = min(15, len(ch_names) - 1)
            ica = mne.preprocessing.ICA(n_components=n_comp, method="fastica",
                                         random_state=42)
            ica.fit(raw, verbose=False)
            fp_cands = [c for c in ch_names if "FP1" in c or "FP2" in c]
            if fp_cands:
                eog_idx, _ = ica.find_bads_eog(raw, ch_name=fp_cands[0], verbose=False)
                ica.exclude = eog_idx[:2]
                ica.apply(raw, verbose=False)
                results["ica_removed"] = list(eog_idx[:2])
            else:
                results["ica_removed"] = []
        except Exception as e:
            results["ica_warning"] = str(e)

    data = raw.get_data()
    results["data"] = data

    progress_cb(40, "Betti-1 …")
    betti_scores = {}
    for region in EEG_REGIONS:
        sig = get_region_signal(data, ch_names, region, sfreq, analysis_duration)
        betti_scores[region] = compute_betti1(sig, sfreq)
    results["betti_scores"] = betti_scores

    progress_cb(55, "Attractors …")
    attractors = {}
    for region in EEG_REGIONS:
        sig = get_region_signal(data, ch_names, region, sfreq, analysis_duration)
        tau = max(1, int(20 * sfreq / 1000))
        X = takens_embed_3d(sig, tau=tau)
        if X is not None and len(X) > 800:
            X = X[np.linspace(0, len(X)-1, 800, dtype=int)]
        attractors[region] = X
    results["attractors"] = attractors

    progress_cb(70, "PLV …")
    plv_series = {}
    for region in EEG_REGIONS:
        sig = get_region_signal(data, ch_names, region, sfreq, analysis_duration)
        plv_series[region] = compute_theta_plv_timeseries(sig, sfreq)
    results["plv_series"] = plv_series

    progress_cb(85, "Coupling …")
    cov, band_names = compute_cross_band_coupling(data, sfreq, analysis_duration)
    results["coupling_matrix"] = cov
    results["band_names"] = band_names

    progress_cb(100, "Done")
    return results


# ── ──────────────────────────────────────────────────────────────────────────
#   MAIN WINDOW
# ── ──────────────────────────────────────────────────────────────────────────

class DeerskinExplorer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deerskin Topological Explorer  ·  PerceptionLab")
        self.setMinimumSize(1200, 820)
        self._results = None
        self._selected_region = "Frontal"
        self._worker = None
        self._folder_worker = None
        # Folder batch state
        self._batch_results = []    # list of (filename, results)
        self._batch_index   = -1    # currently displayed index
        self._setup_style()
        self._build_ui()

    # ── palette / style ──────────────────────────────────────────────────────

    def _setup_style(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background:{BG_DARK}; color:{TEXT_HI}; }}
            QGroupBox {{
                border:1px solid {BORDER};
                border-radius:6px;
                margin-top:6px;
                font-family:'Courier New';
                font-size:10px;
                color:{TEXT_MED};
                padding:4px;
            }}
            QGroupBox::title {{
                subcontrol-origin:margin;
                left:8px;
                padding:0 4px;
                color:{ACCENT1};
            }}
            QPushButton {{
                background:{BG_CARD};
                color:{TEXT_HI};
                border:1px solid {BORDER};
                border-radius:5px;
                padding:6px 14px;
                font-family:'Courier New';
                font-size:10px;
            }}
            QPushButton:hover {{ background:{ACCENT1}22; border-color:{ACCENT1}; }}
            QPushButton:disabled {{ color:{TEXT_DIM}; }}
            QProgressBar {{
                background:{BG_CARD};
                border:1px solid {BORDER};
                border-radius:4px;
                height:10px;
                text-align:center;
                font-size:8px;
                color:{TEXT_MED};
            }}
            QProgressBar::chunk {{ background:{ACCENT1}; border-radius:3px; }}
            QLabel {{ font-family:'Courier New'; }}
            QComboBox {{
                background:{BG_CARD};
                border:1px solid {BORDER};
                color:{TEXT_HI};
                font-family:'Courier New';
                font-size:10px;
                padding:3px;
                border-radius:4px;
            }}
            QComboBox QAbstractItemView {{
                background:{BG_CARD};
                color:{TEXT_HI};
                selection-background-color:{ACCENT1}44;
            }}
            QCheckBox {{ color:{TEXT_MED}; font-family:'Courier New'; font-size:10px; }}
            QCheckBox::indicator:checked {{ background:{ACCENT1}; border:1px solid {ACCENT1}; }}
            QSplitter::handle {{ background:{BORDER}; }}
        """)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_lay = QVBoxLayout(central)
        root_lay.setSpacing(6)
        root_lay.setContentsMargins(8, 8, 8, 8)

        # ── Header ──────────────────────────────────────────────────────────
        hdr = QWidget()
        hdr_lay = QHBoxLayout(hdr)
        hdr_lay.setContentsMargins(0, 0, 0, 0)

        title = QLabel("DEERSKIN TOPOLOGICAL EXPLORER")
        title.setStyleSheet(
            f"color:{ACCENT1}; font-size:16px; font-weight:bold; "
            f"font-family:'Courier New'; letter-spacing:2px;"
        )
        sub = QLabel("oscillatory phase-space geometry · PerceptionLab")
        sub.setStyleSheet(f"color:{TEXT_DIM}; font-size:9px; font-family:'Courier New';")

        title_col = QVBoxLayout()
        title_col.setSpacing(1)
        title_col.addWidget(title)
        title_col.addWidget(sub)
        hdr_lay.addLayout(title_col)
        hdr_lay.addStretch()

        # Controls
        ctrl_box = QGroupBox("Controls")
        ctrl_lay = QHBoxLayout(ctrl_box)
        ctrl_lay.setSpacing(8)

        self.btn_load = QPushButton("⊕  Load EDF")
        self.btn_load.setStyleSheet(
            f"color:{ACCENT1}; border-color:{ACCENT1}; font-weight:bold;"
        )
        self.btn_load.clicked.connect(self._load_edf)
        ctrl_lay.addWidget(self.btn_load)

        self.btn_load_folder = QPushButton("⊞  Load Folder")
        self.btn_load_folder.setStyleSheet(
            f"color:{ACCENT3}; border-color:{ACCENT3}; font-weight:bold;"
        )
        self.btn_load_folder.clicked.connect(self._load_folder)
        ctrl_lay.addWidget(self.btn_load_folder)

        self.chk_ica = QCheckBox("ICA artifact rejection")
        self.chk_ica.setChecked(True)
        ctrl_lay.addWidget(self.chk_ica)

        self.cmb_region = QComboBox()
        self.cmb_region.addItems(list(EEG_REGIONS.keys()))
        self.cmb_region.currentTextChanged.connect(self._on_region_changed)
        ctrl_lay.addWidget(QLabel("Region:"))
        ctrl_lay.addWidget(self.cmb_region)

        self.btn_analyze = QPushButton("▶  Analyze")
        self.btn_analyze.setEnabled(False)
        self.btn_analyze.setStyleSheet(
            f"color:{ACCENT4}; border-color:{ACCENT4}; font-weight:bold;"
        )
        self.btn_analyze.clicked.connect(self._run_analysis)
        ctrl_lay.addWidget(self.btn_analyze)

        hdr_lay.addWidget(ctrl_box)
        root_lay.addWidget(hdr)

        # ── Progress ────────────────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet(
            f"color:{TEXT_DIM}; font-size:8px; font-family:'Courier New';"
        )
        prog_row = QHBoxLayout()
        prog_row.addWidget(self.progress_bar)
        prog_row.addWidget(self.progress_label)
        root_lay.addLayout(prog_row)

        # ── Four panels ─────────────────────────────────────────────────────
        panels_splitter = QSplitter(Qt.Horizontal)

        self.panel_a = PanelA()
        self.panel_a.region_clicked.connect(self._on_region_click)
        panels_splitter.addWidget(self._wrap_panel(self.panel_a))

        self.panel_b = PanelB()
        panels_splitter.addWidget(self._wrap_panel(self.panel_b))

        right_splitter = QSplitter(Qt.Vertical)
        self.panel_c = PanelC()
        right_splitter.addWidget(self._wrap_panel(self.panel_c))
        self.panel_d = PanelD()
        right_splitter.addWidget(self._wrap_panel(self.panel_d))
        panels_splitter.addWidget(right_splitter)

        panels_splitter.setSizes([310, 310, 540])
        root_lay.addWidget(panels_splitter, stretch=8)

        # ── Bottom bar: nav + tabbed log/analysis ────────────────────────────
        bottom_bar = QWidget()
        bottom_lay = QHBoxLayout(bottom_bar)
        bottom_lay.setContentsMargins(0, 0, 0, 0)
        bottom_lay.setSpacing(6)

        # Back / Forward navigation (folder mode)
        nav_box = QGroupBox("Batch Navigation")
        nav_lay = QHBoxLayout(nav_box)
        nav_lay.setSpacing(4)

        self.btn_prev = QPushButton("◀  Prev")
        self.btn_prev.setEnabled(False)
        self.btn_prev.setFixedWidth(80)
        self.btn_prev.clicked.connect(self._batch_prev)
        nav_lay.addWidget(self.btn_prev)

        self.lbl_batch_pos = QLabel("—")
        self.lbl_batch_pos.setStyleSheet(
            f"color:{TEXT_MED}; font-size:9px; font-family:'Courier New';"
        )
        self.lbl_batch_pos.setAlignment(Qt.AlignCenter)
        self.lbl_batch_pos.setFixedWidth(120)
        nav_lay.addWidget(self.lbl_batch_pos)

        self.btn_next = QPushButton("Next  ▶")
        self.btn_next.setEnabled(False)
        self.btn_next.setFixedWidth(80)
        self.btn_next.clicked.connect(self._batch_next)
        nav_lay.addWidget(self.btn_next)

        self.btn_export = QPushButton("⬇  Export JSON")
        self.btn_export.setEnabled(False)
        self.btn_export.setStyleSheet(
            f"color:{ACCENT2}; border-color:{ACCENT2};"
        )
        self.btn_export.clicked.connect(self._export_json)
        nav_lay.addWidget(self.btn_export)

        bottom_lay.addWidget(nav_box)

        # Tab: Log | Analysis
        self.bottom_tabs = QTabWidget()
        self.bottom_tabs.setStyleSheet(
            f"QTabWidget::pane {{ border:1px solid {BORDER}; background:{BG_PANEL}; }}"
            f"QTabBar::tab {{ background:{BG_CARD}; color:{TEXT_DIM}; "
            f"font-family:'Courier New'; font-size:9px; padding:3px 8px; }}"
            f"QTabBar::tab:selected {{ color:{ACCENT1}; border-bottom:2px solid {ACCENT1}; }}"
        )

        self.log_panel = LogPanel()
        self.bottom_tabs.addTab(self.log_panel, "Log")

        self.analysis_panel = AnalysisPanel()
        self.bottom_tabs.addTab(self.analysis_panel, "Group Analysis (HC vs SZ)")

        bottom_lay.addWidget(self.bottom_tabs, stretch=1)

        root_lay.addWidget(bottom_bar)
        bottom_bar.setFixedHeight(115)

        # ── Status bar ──────────────────────────────────────────────────────
        sb = QStatusBar()
        sb.setStyleSheet(f"color:{TEXT_DIM}; font-family:'Courier New'; font-size:9px;")
        sb.showMessage("Ready. Load an EDF file or folder to begin.")
        self.setStatusBar(sb)
        self._sb = sb

    def _wrap_panel(self, widget):
        frame = QFrame()
        frame.setStyleSheet(
            f"QFrame {{ border:1px solid {BORDER}; border-radius:6px; "
            f"background:{BG_PANEL}; }}"
        )
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.addWidget(widget)
        return frame

    # ── File loading ─────────────────────────────────────────────────────────

    def _load_edf(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open EDF File", "", "EDF Files (*.edf *.EDF)"
        )
        if not path:
            return
        self._filepath = path
        fname = path.replace("\\", "/").split("/")[-1]
        self.btn_analyze.setEnabled(True)
        self._sb.showMessage(f"Loaded: {fname}  —  click Analyze to begin")
        self.log_panel.log(f"File selected: {fname}")

    def _load_folder(self):
        import os
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing EDF Files", ""
        )
        if not folder:
            return
        edfs = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".edf")
        ])
        if not edfs:
            self.log_panel.log("No EDF files found in selected folder.")
            return

        self.log_panel.clear()
        self.log_panel.log(f"Folder: {folder}")
        self.log_panel.log(f"Found {len(edfs)} EDF file(s). Starting batch analysis …")
        self._batch_results = []
        self._batch_index   = -1
        self._update_nav_buttons()

        self.btn_load.setEnabled(False)
        self.btn_load_folder.setEnabled(False)
        self.btn_analyze.setEnabled(False)

        self._folder_worker = FolderWorker(
            edfs, self.chk_ica.isChecked(), 60.0
        )
        self._folder_worker.file_started.connect(self._on_folder_file_started)
        self._folder_worker.file_done.connect(self._on_folder_file_done)
        self._folder_worker.file_error.connect(self._on_folder_file_error)
        self._folder_worker.all_done.connect(self._on_folder_all_done)
        self._folder_worker.sub_progress.connect(
            lambda p, m: (self.progress_bar.setValue(p),
                          self.progress_label.setText(m))
        )
        self._folder_worker.start()

    # ── Analysis ─────────────────────────────────────────────────────────────

    def _run_analysis(self):
        if not hasattr(self, "_filepath"):
            return
        self.btn_analyze.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.log_panel.clear()
        self.log_panel.log("Starting analysis pipeline …")

        self._worker = AnalysisWorker(
            self._filepath,
            ica_enabled=self.chk_ica.isChecked(),
            analysis_duration=60.0,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.progress_label.setText(msg)
        self.log_panel.log(msg)
        self._sb.showMessage(msg)

    def _on_finished(self, results: dict):
        self._results = results
        self.btn_analyze.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_label.setText("Done")

        if "ica_removed" in results:
            n = len(results["ica_removed"])
            self.log_panel.log(f"ICA: removed {n} eye-movement component(s).")
        if "ica_warning" in results:
            self.log_panel.log(f"ICA warning: {results['ica_warning']}")

        # Panel A
        self.panel_a.update_scores(results["betti_scores"])

        # Log Betti scores
        for region, score in results["betti_scores"].items():
            self.log_panel.log(f"Betti-1 [{region}]: {score:.4f}")

        # Panel D
        self.panel_d.update_coupling(
            results["coupling_matrix"], results["band_names"]
        )

        # Update region panels for current selection
        self._refresh_region_panels()

        self._sb.showMessage("Analysis complete. Click a brain region to inspect.")

    def _on_error(self, msg: str):
        self.btn_analyze.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.log_panel.log(f"ERROR: {msg}")
        self._sb.showMessage("Error during analysis. See log.")

    # ── Folder batch callbacks ────────────────────────────────────────────────

    def _on_folder_file_started(self, idx: int, total: int, fname: str):
        self.log_panel.log(f"[{idx+1}/{total}] Analysing {fname} …")
        self._sb.showMessage(f"Batch {idx+1}/{total}: {fname}")
        pct = int(100 * idx / total)
        self.progress_bar.setValue(pct)

    def _on_folder_file_done(self, idx: int, fname: str, results: dict):
        self._batch_results.append((fname, results))
        b = results.get("betti_scores", {})
        line = "  β₁  " + "  ".join(f"{r}:{v:.2f}" for r, v in b.items())
        ica_n = len(results.get("ica_removed", []))
        self.log_panel.log(f"  ✓ {fname}  ICA-{ica_n}  {line}")
        # Auto-display first result
        if len(self._batch_results) == 1:
            self._batch_index = 0
            self._display_batch_entry(0)
        self._update_nav_buttons()

    def _on_folder_file_error(self, idx: int, fname: str, err: str):
        self._batch_results.append((fname, {"error": err}))
        self.log_panel.log(f"  ✗ {fname}  ERROR: {err.splitlines()[0]}")
        self._update_nav_buttons()

    def _on_folder_all_done(self, completed: list):
        self.btn_load.setEnabled(True)
        self.btn_load_folder.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.progress_bar.setValue(100)
        self.progress_label.setText("Batch complete")
        self.log_panel.log(f"═══ Batch complete: {len(completed)} files processed. ═══")
        self._update_nav_buttons()
        self._sb.showMessage(
            f"Batch done — {len(completed)} files. Use ◀ ▶ to browse. See Group Analysis tab."
        )
        # Run statistics and show analysis panel
        self.analysis_panel.log_fn = self.log_panel.log
        self.analysis_panel.run(self._batch_results)
        self.bottom_tabs.setCurrentIndex(1)  # switch to analysis tab

    # ── Navigation ────────────────────────────────────────────────────────────

    def _update_nav_buttons(self):
        n = len(self._batch_results)
        i = self._batch_index
        self.btn_prev.setEnabled(n > 0 and i > 0)
        self.btn_next.setEnabled(n > 0 and i < n - 1)
        if n > 0 and i >= 0:
            fname = self._batch_results[i][0]
            self.lbl_batch_pos.setText(f"{i+1}/{n}  {fname[:18]}")
        else:
            self.lbl_batch_pos.setText(f"0/{n}" if n else "—")

    def _batch_prev(self):
        if self._batch_index > 0:
            self._batch_index -= 1
            self._display_batch_entry(self._batch_index)
            self._update_nav_buttons()

    def _batch_next(self):
        if self._batch_index < len(self._batch_results) - 1:
            self._batch_index += 1
            self._display_batch_entry(self._batch_index)
            self._update_nav_buttons()

    def _display_batch_entry(self, idx: int):
        fname, results = self._batch_results[idx]
        if "error" in results:
            self.log_panel.log(f"[view] {fname}: error — {results['error'][:80]}")
            return
        self._results = results
        self.log_panel.log(f"[view {idx+1}] {fname}")
        self.panel_a.update_scores(results.get("betti_scores", {}))
        self.panel_d.update_coupling(
            results.get("coupling_matrix"), results.get("band_names", [])
        )
        self._refresh_region_panels()
        self._sb.showMessage(f"Viewing {fname}  ({idx+1}/{len(self._batch_results)})")

    # ── JSON export ───────────────────────────────────────────────────────────

    def _export_json(self):
        import json, os
        if not self._batch_results:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Batch Results", "deerskin_batch_results.json",
            "JSON Files (*.json)"
        )
        if not path:
            return

        export = []
        for fname, results in self._batch_results:
            entry = {"file": fname}
            if "error" in results:
                entry["error"] = results["error"]
                export.append(entry)
                continue
            entry["betti_scores"] = {
                k: float(v) for k, v in results.get("betti_scores", {}).items()
            }
            entry["ica_removed_count"] = len(results.get("ica_removed", []))
            plv = results.get("plv_series", {})
            entry["theta_plv"] = {
                region: {
                    "mean": float(np.mean(v)) if v is not None and len(v) else None,
                    "std":  float(np.std(v))  if v is not None and len(v) else None,
                }
                for region, v in plv.items()
            }
            mat = results.get("coupling_matrix")
            bands = results.get("band_names", [])
            if mat is not None and len(bands):
                if isinstance(mat, dict):
                    entry["coupling_matrix"] = mat
                    off_vals = [abs(mat[b1][b2])
                                for b1 in bands for b2 in bands if b1 != b2]
                    entry["mean_cross_band_coupling"] = float(
                        np.mean(off_vals)) if off_vals else 0.0
                else:
                    off = [(i, j) for i in range(len(bands))
                           for j in range(len(bands)) if i != j]
                    entry["mean_cross_band_coupling"] = float(
                        np.mean([abs(mat[i, j]) for i, j in off])
                    )
                    entry["coupling_matrix"] = {
                        bands[i]: {bands[j]: float(mat[i, j])
                                   for j in range(len(bands))}
                        for i in range(len(bands))
                    }
                if "coupling_matrix_dict" in results:
                    entry["coupling_matrix"] = results["coupling_matrix_dict"]
            export.append(entry)

        with open(path, "w") as f:
            json.dump(export, f, indent=2)
        self.log_panel.log(f"Exported {len(export)} records → {path}")
        self._sb.showMessage(f"JSON exported: {os.path.basename(path)}")

    # ── Region selection ─────────────────────────────────────────────────────

    def _on_region_click(self, region: str):
        self._selected_region = region
        self.cmb_region.blockSignals(True)
        self.cmb_region.setCurrentText(region)
        self.cmb_region.blockSignals(False)
        self._refresh_region_panels()

    def _on_region_changed(self, region: str):
        self._selected_region = region
        self._refresh_region_panels()

    def _refresh_region_panels(self):
        if self._results is None:
            return
        region = self._selected_region

        # Panel B
        attractor = self._results["attractors"].get(region)
        self.panel_b.set_attractor(attractor, region)

        # Panel C
        plv = self._results["plv_series"].get(region)
        self.panel_c.update_plv(plv, region)

        self.log_panel.log(f"Region selected: {region}")
        if plv is not None and len(plv) > 0:
            self.log_panel.log(
                f"  Theta PLV — μ={float(np.mean(plv)):.3f}  σ={float(np.std(plv)):.4f}"
            )


# ── ──────────────────────────────────────────────────────────────────────────
#   ENTRY POINT
# ── ──────────────────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Deerskin Topological Explorer")
    app.setStyle("Fusion")

    # Dark palette for native widgets
    palette = QPalette()
    palette.setColor(QPalette.Window,       QColor(BG_DARK))
    palette.setColor(QPalette.WindowText,   QColor(TEXT_HI))
    palette.setColor(QPalette.Base,         QColor(BG_PANEL))
    palette.setColor(QPalette.Text,         QColor(TEXT_HI))
    palette.setColor(QPalette.Button,       QColor(BG_CARD))
    palette.setColor(QPalette.ButtonText,   QColor(TEXT_HI))
    palette.setColor(QPalette.Highlight,    QColor(ACCENT1))
    palette.setColor(QPalette.HighlightedText, QColor(BG_DARK))
    app.setPalette(palette)

    win = DeerskinExplorer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()