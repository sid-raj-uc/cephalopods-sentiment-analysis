"""
Cephalopod Behavioral Sentiment Analysis — GSoC 2026 Entry Task

Extracts two features from a video clip:
  1. Frame-to-frame motion magnitude (frame differencing)
  2. Inter-frame histogram change (chromatophore activity proxy)

Saves extracted features to a .npz file so notebooks can skip re-processing.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d


def load_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    meta = {
        "fps": fps,
        "width": width,
        "height": height,
        "n_frames": len(frames),
        "duration": len(frames) / fps if fps > 0 else 0.0,
    }
    return frames, meta


def compute_motion_magnitude(frames):
    magnitudes = np.empty(len(frames) - 1)
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    for i, frame in enumerate(frames[1:]):
        curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        magnitudes[i] = np.mean(np.abs(curr - prev))
        prev = curr
    return magnitudes


def compute_histogram_changes(frames):
    def frame_hist(f):
        hist = np.zeros(256 * 3, dtype=np.float32)
        for c in range(3):
            h = cv2.calcHist([f], [c], None, [256], [0, 256]).flatten()
            hist[c * 256:(c + 1) * 256] = h
        cv2.normalize(hist, hist)
        return hist

    changes = np.empty(len(frames) - 1)
    prev_h = frame_hist(frames[0])
    for i, frame in enumerate(frames[1:]):
        curr_h = frame_hist(frame)
        corr = cv2.compareHist(prev_h, curr_h, cv2.HISTCMP_CORREL)
        changes[i] = 1.0 - float(np.clip(corr, -1.0, 1.0))
        prev_h = curr_h
    return changes


def smooth(signal, fps, window_sec=0.4):
    w = max(1, int(round(fps * window_sec)))
    return uniform_filter1d(signal, size=w)


def norm01(x):
    r = x.max() - x.min()
    return (x - x.min()) / r if r > 0 else np.zeros_like(x)


def activity_spans(signal, times, threshold):
    spans = []
    in_span, t0 = False, 0.0
    for i, flag in enumerate(signal > threshold):
        if flag and not in_span:
            in_span, t0 = True, times[i]
        elif not flag and in_span:
            spans.append((t0, times[i]))
            in_span = False
    if in_span:
        spans.append((t0, times[-1]))
    return spans


def visualize(motion, hist_ch, fps, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    times = np.arange(len(motion)) / fps
    sm_motion = smooth(motion, fps)
    sm_hist = smooth(hist_ch, fps)
    score = 0.5 * norm01(sm_motion) + 0.5 * norm01(sm_hist)

    thr_motion = sm_motion.mean() + 0.5 * sm_motion.std()
    thr_hist = sm_hist.mean() + 0.5 * sm_hist.std()
    thr_score = score.mean() + 0.4 * score.std()

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Cephalopod Behavioral Analysis", fontsize=14, fontweight="bold", y=0.98)

    ax = axes[0]
    ax.fill_between(times, sm_motion, alpha=0.25, color="#1f77b4")
    ax.plot(times, sm_motion, color="#1f77b4", linewidth=1.2, label="Motion magnitude")
    ax.axhline(thr_motion, color="#1f77b4", linestyle="--", linewidth=0.8, alpha=0.7)
    for t0, t1 in activity_spans(sm_motion, times, thr_motion):
        ax.axvspan(t0, t1, color="#1f77b4", alpha=0.12)
    ax.set_ylabel("Mean |ΔPixel|", fontsize=10)
    ax.set_title("Feature 1 — Frame-to-Frame Motion Magnitude", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.fill_between(times, sm_hist, alpha=0.25, color="#ff7f0e")
    ax.plot(times, sm_hist, color="#ff7f0e", linewidth=1.2, label="Histogram change")
    ax.axhline(thr_hist, color="#ff7f0e", linestyle="--", linewidth=0.8, alpha=0.7)
    for t0, t1 in activity_spans(sm_hist, times, thr_hist):
        ax.axvspan(t0, t1, color="#ff7f0e", alpha=0.12)
    ax.set_ylabel("1 − Hist. Correlation", fontsize=10)
    ax.set_title("Feature 2 — Inter-Frame Histogram Change (Chromatophore Activity)", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.fill_between(times, score, alpha=0.3, color="#2ca02c")
    ax.plot(times, score, color="#2ca02c", linewidth=1.4, label="Activity score")
    ax.axhline(thr_score, color="#2ca02c", linestyle="--", linewidth=0.8, alpha=0.7)
    for t0, t1 in activity_spans(score, times, thr_score):
        ax.axvspan(t0, t1, color="#2ca02c", alpha=0.15)
    ax.legend(
        handles=[
            mpatches.Patch(color="#2ca02c", linewidth=1.4, label="Activity score"),
            mpatches.Patch(color="#2ca02c", alpha=0.15, label="High-activity period"),
        ],
        loc="upper right", fontsize=8,
    )
    ax.set_ylabel("Normalised score", fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_title("Combined Behavioral Activity Score", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {output_path}")


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "data/data_octopus.mp4"
    clip_name = os.path.splitext(os.path.basename(video_path))[0]
    plot_path = os.path.join("plots", clip_name, "behavioral_analysis.png")
    features_path = os.path.join("data", f"{clip_name}_features.npz")

    print(f"\nLoading: {video_path}")
    frames, meta = load_video(video_path)
    fps = meta["fps"]
    print(f"  {meta['width']}x{meta['height']}  {fps:.2f} fps  {meta['n_frames']} frames  {meta['duration']:.1f}s")

    print("Extracting motion magnitude...")
    motion = compute_motion_magnitude(frames)

    print("Extracting histogram changes...")
    hist_ch = compute_histogram_changes(frames)

    print(f"\nMotion   — mean: {motion.mean():.3f}  std: {motion.std():.3f}  max: {motion.max():.3f}")
    print(f"Hist-chg — mean: {hist_ch.mean():.4f}  std: {hist_ch.std():.4f}  max: {hist_ch.max():.4f}")

    np.savez(features_path,
             motion=motion, hist_ch=hist_ch,
             fps=np.array(fps),
             width=np.array(meta["width"]), height=np.array(meta["height"]),
             n_frames=np.array(meta["n_frames"]), duration=np.array(meta["duration"]))
    print(f"\nFeatures saved → {features_path}")

    visualize(motion, hist_ch, fps, output_path=plot_path)


if __name__ == "__main__":
    main()
