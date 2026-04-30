"""
dct_analysis.py — Discrete Cosine Transform (DCT) frequency analysis.

Generates:
  1. Log-magnitude DCT spectrum heatmap (full 2-D frequency map)
  2. Frequency band energy bar visualisation (low / mid / high bands)
  3. Summary statistics for the LLM prompt
"""
from __future__ import annotations

import base64
import io
import logging

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from scipy.fft import dctn
logger = logging.getLogger("agrixai.dct")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_gray(original_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 → float64 grayscale in [0, 1]."""
    gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float64) / 255.0


def _band_energy(dct_shifted: np.ndarray, r_inner: float, r_outer: float) -> float:
    """Sum energy of DCT coefficients within a radial band."""
    h, w = dct_shifted.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    r_max = min(cy, cx)
    mask = (dist >= r_inner * r_max) & (dist < r_outer * r_max)
    return float(np.sum(dct_shifted[mask] ** 2))


# ── Public API ────────────────────────────────────────────────────────────────

def analyse(original_rgb: np.ndarray) -> dict:
    """
    Run 2-D DCT analysis on an RGB image.

    Returns a dict with:
        spectrum_b64   : base64 PNG — log-magnitude DCT spectrum heatmap
        band_chart_b64 : base64 PNG — bar chart of low/mid/high energy
        stats          : dict with numeric frequency stats
    """
    logger.info("Starting DCT analysis...")
    gray = _to_gray(original_rgb)

    MAX_DCT_SIZE = 512
    h, w = gray.shape
    if max(h, w) > MAX_DCT_SIZE:
        scale = MAX_DCT_SIZE / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        logger.info("Resized image from %dx%d to %dx%d for DCT analysis", w, h, gray.shape[1], gray.shape[0])

    # ── 2-D DCT ───────────────────────────────────────────────────────────
    dct_coeffs = dctn(gray, norm="ortho")               # same shape as gray

    # Shift DC component to centre (like np.fft.fftshift)
    dct_shifted = np.fft.fftshift(dct_coeffs)
    log_mag_shifted = np.log1p(np.abs(dct_shifted))

    # Normalise for display
    vmax = np.percentile(log_mag_shifted, 99.0)
    log_mag_norm = np.clip(log_mag_shifted, 0, vmax) / vmax

    bands  = ["Low\n(texture/structure)", "Mid\n(edges/detail)", "High\n(noise/fine detail)"]
    fig1 = fig2 = None
    try:
        # ── Spectrum heatmap ──────────────────────────────────────────────────
        fig1 = Figure(figsize=(6, 5), facecolor="#0a0f1e")
        ax1 = fig1.add_subplot(111, facecolor="#0a0f1e")
        
        im = ax1.imshow(log_mag_norm, cmap="inferno", interpolation="bilinear")
        ax1.set_title("DCT Frequency Spectrum", color="#10b981", fontsize=13, fontweight="bold")
        ax1.axis("off")
        cbar = fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white")
        cbar.ax.set_yticklabels(cbar.ax.get_yticks(), color="white", fontsize=8)
        cbar.set_label("Relative Energy Magnitude", color="#06b6d4", fontsize=9)
        fig1.tight_layout(pad=0.1)

        buf = io.BytesIO()
        fig1.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0a0f1e")
        spectrum_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # ── Frequency band energies ────────────────────────────────────────────
        low_e  = _band_energy(dct_shifted, 0.0,  0.15)
        mid_e  = _band_energy(dct_shifted, 0.15, 0.45)
        high_e = _band_energy(dct_shifted, 0.45, 1.0)
        total  = low_e + mid_e + high_e + 1e-12
        low_pct  = low_e  / total * 100
        mid_pct  = mid_e  / total * 100
        high_pct = high_e / total * 100

        # ── Band energy bar chart ─────────────────────────────────────────────
        fig2 = Figure(figsize=(5, 3.2), facecolor="#0a0f1e")
        ax2 = fig2.add_subplot(111, facecolor="#111827")
        values = [low_pct, mid_pct, high_pct]
        colors = ["#10b981", "#06b6d4", "#8b5cf6"]
        
        bars = ax2.bar(bands, values, color=colors, width=0.5, zorder=3)
        for bar, val in zip(bars, values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val:.1f}%",
                ha="center", va="bottom", color="white", fontsize=10, fontweight="bold",
            )
        ax2.set_ylim(0, max(values) * 1.25)
        ax2.set_ylabel("Energy %", color="#9ca3af", fontsize=9)
        ax2.set_title("Frequency Band Energy Distribution", color="#10b981",
                      fontsize=11, fontweight="bold")
        ax2.tick_params(colors="white")
        ax2.spines["top"].set_color("#374151")
        ax2.spines["right"].set_color("#374151")
        ax2.spines["bottom"].set_color("#374151")
        ax2.spines["left"].set_color("#374151")
        ax2.yaxis.label.set_color("#9ca3af")
        for label in ax2.get_xticklabels():
            label.set_color("#d1d5db")
        ax2.grid(axis="y", color="#374151", linestyle="--", alpha=0.5, zorder=0)
        fig2.tight_layout(pad=0.8)

        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png", dpi=120, bbox_inches="tight", facecolor="#0a0f1e")
        band_chart_b64 = base64.b64encode(buf2.getvalue()).decode("utf-8")

    finally:
        # Prevent memory leaks with Agg backend
        if fig1:
            plt.close(fig1)
        if fig2:
            plt.close(fig2)

    # ── DC coefficient (mean brightness) ──────────────────────────────────
    dc_val = float(dct_coeffs[0, 0])
    dominant_band = bands[int(np.argmax(values))].split("\n")[0]

    stats = {
        "low_energy_pct":  round(low_pct, 2),
        "mid_energy_pct":  round(mid_pct, 2),
        "high_energy_pct": round(high_pct, 2),
        "dc_coefficient":  round(dc_val, 4),
        "dominant_band":   dominant_band,
    }

    logger.info("DCT analysis complete.")
    return {
        "spectrum_b64":    spectrum_b64,
        "band_chart_b64":  band_chart_b64,
        "stats":           stats,
    }
