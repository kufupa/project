#!/usr/bin/env python3
"""Build thesis figures from on-disk eval JSON (no training jobs)."""
from __future__ import annotations

import json
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
PROJECT = ROOT.parent
ART = PROJECT / "artifacts"
EPH = Path("/rds/general/ephemeral/user/aa6622/ephemeral/smolvla_metaworld")

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

COLORS = {
    "grpo": "#2563eb",
    "direct": "#059669",
    "eggroll": "#d97706",
    "wm": "#7c3aed",
    "baseline": "#6b7280",
    "bounded": "#059669",
    "raw": "#dc2626",
}


def load_sweep(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    rows = data.get("rows", [])
    return sorted(rows, key=lambda r: int(r["update"]))


def merge_sweeps(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    pts: dict[int, float] = {}
    for p in paths:
        if not p.exists():
            continue
        for row in load_sweep(p):
            pts[int(row["update"])] = float(row["pc_success"])
    if not pts:
        return np.array([]), np.array([])
    xs = np.array(sorted(pts))
    ys = np.array([pts[x] for x in xs])
    return xs, ys


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / name
    fig.savefig(out, format="pdf")
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    print(f"wrote {out}")


def plot_line(
    xs: np.ndarray,
    ys: np.ndarray,
    title: str,
    fname: str,
    ylabel: str = "Held-out success (%)",
    xlabel: str = "Training update",
    color: str = COLORS["grpo"],
    baseline: float | None = 21.0,
    mark_best: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(xs, ys, "-o", color=color, markersize=4, linewidth=1.5)
    if baseline is not None:
        ax.axhline(baseline, color=COLORS["baseline"], linestyle="--", linewidth=1, label=f"SmolVLA baseline ({baseline:.0f}%)")
    if mark_best and len(ys):
        i = int(np.argmax(ys))
        ax.scatter([xs[i]], [ys[i]], s=60, color=color, edgecolors="black", linewidths=0.6, zorder=5)
        ax.annotate(
            f"peak {ys[i]:.0f}% @ u={int(xs[i])}",
            (xs[i], ys[i]),
            textcoords="offset points",
            xytext=(6, 8),
            fontsize=8,
        )
    if len(ys):
        ax.scatter([xs[-1]], [ys[-1]], s=40, facecolors="white", edgecolors=color, linewidths=1.2, zorder=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, max(55, float(np.max(ys)) + 8) if len(ys) else 55)
    ax.grid(True, alpha=0.25)
    if baseline is not None:
        ax.legend(loc="upper right", frameon=True)
    save(fig, fname)


def fig_grpo_g16_long() -> None:
    paths = [
        ART / "phase11_pushv3_chunk5_g16_lr5e6_clip02_u20/eval_sweep_0002_0010_25ep_nenv25_async/eval_sweep_summary.json",
        ART / "phase11_pushv3_chunk5_g16_lr5e6_clip02_u20/eval_sweep_0012_0020_25ep_nenv25_async/eval_sweep_summary.json",
        EPH / "phase11_pushv3_chunk5_g16_lr5e6_clip02_u70/eval25_stride2_0022_0070/eval_sweep_summary.json",
        EPH / "phase11_pushv3_chunk5_g16_lr5e6_clip02_u70/eval25_stride2_0072_0170/eval_sweep_summary.json",
    ]
    xs, ys = merge_sweeps(paths)
    plot_line(
        xs,
        ys,
        "Environment-reward GRPO (group 16): held-out success vs update",
        "fig_grpo_g16_25ep_curve.pdf",
        color=COLORS["grpo"],
    )


def fig_grpo_g8_collapse() -> None:
    p = ART / "phase11_pushv3_chunk5_g8_vecasync_u100/eval_sweep_0005_0050_25ep_nenv25_async/eval_sweep_summary.json"
    xs, ys = merge_sweeps([p])
    plot_line(
        xs,
        ys,
        "Environment-reward GRPO (group 8): 25-episode held-out sweep",
        "fig_grpo_g8_25ep_collapse.pdf",
        color=COLORS["grpo"],
    )


def fig_wm_collapse() -> None:
    base = ART / "phase12_wm_only_overnight_70u_20260519_021516_telemetryfix/official_g8_lr1e5"
    paths = sorted(base.glob("eval_last5_*_25ep_nenv25/eval_sweep_summary.json"))
    xs, ys = merge_sweeps(paths)
    plot_line(
        xs,
        ys,
        "World-model-reward GRPO: held-out push-v3 success vs update",
        "fig_wm_reward_25ep_curve.pdf",
        color=COLORS["wm"],
        baseline=21.0,
    )


def fig_eval_horizon_compare() -> None:
    """Same checkpoint, different eval episode counts (group-8 u10)."""
    labels = ["Baseline\n(100 ep)", "GRPO u10\n(25 ep)", "GRPO u10\n(100 ep)", "GRPO u15\n(100 ep)"]
    values = [21.0, 48.0, 33.0, 29.0]
    colors = [COLORS["baseline"], COLORS["grpo"], COLORS["grpo"], COLORS["grpo"]]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.4)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Held-out success (%)")
    ax.set_title("Short vs long held-out evaluation (same policy family)")
    ax.set_ylim(0, 58)
    ax.grid(True, axis="y", alpha=0.25)
    save(fig, "fig_eval_horizon_compare.pdf")


def fig_headline_bars() -> None:
    labels = [
        "SmolVLA\nbaseline",
        "GRPO G8\n100 ep",
        "GRPO G8\n25 ep peak",
        "GRPO G16\n25 ep peak",
        "Direct PPO\n25 ep peak",
        "WM-only\n25 ep peak",
    ]
    values = [21, 33, 48, 52, 40, 36]
    confirmed = [True, True, False, False, False, False]
    colors = []
    for i, ok in enumerate(confirmed):
        if i == 0:
            colors.append(COLORS["baseline"])
        elif labels[i].startswith("Direct"):
            colors.append(COLORS["direct"])
        elif labels[i].startswith("WM"):
            colors.append(COLORS["wm"])
        else:
            colors.append(COLORS["grpo"] if ok else "#93c5fd")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.4)
    for bar, val, ok in zip(bars, values, confirmed):
        tag = f"{val:.0f}%"
        if not ok and val > 21:
            tag += "*"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, tag, ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Held-out success (%)")
    ax.set_title("Headline push-v3 results (* = 25-episode sweep only)")
    ax.set_ylim(0, 58)
    ax.grid(True, axis="y", alpha=0.25)
    save(fig, "fig_headline_bars.pdf")


def fig_direct_sparse() -> None:
    """Sparse checkpoints from completed Direct stage3b eval (handoff table)."""
    pts = {
        0: 16,
        60: 36,
        120: 40,
        140: 36,
        200: 32,
        220: 24,
        240: 28,
    }
    xs = np.array(sorted(pts))
    ys = np.array([pts[k] for k in xs])
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(xs, ys, "-o", color=COLORS["direct"], markersize=5, linewidth=1.5)
    ax.axhline(21, color=COLORS["baseline"], linestyle="--", linewidth=1, label="SmolVLA baseline (21%)")
    i = int(np.argmax(ys))
    ax.scatter([xs[i]], [ys[i]], s=60, color=COLORS["direct"], edgecolors="black", linewidths=0.6)
    ax.annotate(f"peak {ys[i]:.0f}% @ u={int(xs[i])}", (xs[i], ys[i]), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("Training update")
    ax.set_ylabel("Held-out success (%)")
    ax.set_title("Direct sparse PPO: selected 25-episode checkpoints")
    ax.set_ylim(0, 48)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "fig_direct_ppo_25ep_sparse.pdf")


def fig_mt57_aggregate() -> None:
    summary = json.loads(
        (ART / "MT50_Phase57_raw_vs_bounded_decode_25ep_s1000_max180_5x1gpu/phase57_mt50_summary.json").read_text()
    )
    labels = ["Bounded closer", "Raw closer", "Tied columns"]
    values = [
        100 * summary["bounded_win_fraction"],
        100 * summary["raw_win_fraction"],
        100 * (1.0 - summary["bounded_win_fraction"] - summary["raw_win_fraction"]),
    ]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    cols = [COLORS["bounded"], COLORS["raw"], COLORS["baseline"]]
    bars = ax.bar(labels, values, color=cols, edgecolor="black", linewidth=0.4)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{val:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("Share of latent comparison columns (%)")
    ax.set_title("MT50 audit: which action branch is closer to real latents?")
    ax.set_ylim(0, 75)
    ax.grid(True, axis="y", alpha=0.25)
    save(fig, "fig_mt57_latent_column_wins.pdf")


def fig_mt50_success_difficulty() -> None:
    data = json.loads(
        (
            ART
            / "MT50_Phase27_smolvla_baseline_official_lerobot_25ep_s1000_4gpu_rtx6000/phase27_hybrid_50task_success.json"
        ).read_text()
    )
    order = ["easy", "medium", "hard", "very_hard"]
    labels = ["Easy", "Medium", "Hard", "Very hard"]
    vals = []
    for d in order:
        row = next(r for r in data["by_difficulty"] if r["difficulty"] == d)
        vals.append(row["pc_success_micro_episode_weighted"])
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.bar(labels, vals, color=COLORS["baseline"], edgecolor="black", linewidth=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("Episode success rate (%)")
    ax.set_title("Unfine-tuned SmolVLA on MT50 (25 ep per task)")
    ax.set_ylim(0, 85)
    ax.grid(True, axis="y", alpha=0.25)
    save(fig, "fig_mt50_baseline_by_difficulty.pdf")


def fig_hypothesis_verdict() -> None:
  # small schematic: not needed as separate file; table in LaTeX
    pass


def main() -> None:
    fig_grpo_g16_long()
    fig_grpo_g8_collapse()
    fig_wm_collapse()
    fig_eval_horizon_compare()
    fig_headline_bars()
    fig_direct_sparse()
    fig_mt57_aggregate()
    fig_mt50_success_difficulty()
    print("done")


if __name__ == "__main__":
    main()
