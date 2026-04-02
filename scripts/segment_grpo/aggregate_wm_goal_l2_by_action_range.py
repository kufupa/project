#!/usr/bin/env python3
"""Aggregate WM--goal L2 (integer-rounded per megastep) across episodes by env action range.

Reads segment_grpo / out_episode JSON artifacts. Optional second run for paired Policy vs oracle-replay.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BinKey:
    lo: int
    hi: int

    def label(self) -> str:
        return f"{self.lo}:{self.hi}"

    def latex_range(self) -> str:
        return f"{self.lo}--{self.hi}"


def _latex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def _infer_glob(run_dir: Path) -> str:
    if any(run_dir.glob("out_episode_*.json")):
        return "out_episode_*.json"
    if any(run_dir.glob("segment_grpo_episode_*.json")):
        return "segment_grpo_episode_*.json"
    raise FileNotFoundError(f"No out_episode_*.json or segment_grpo_episode_*.json under {run_dir}")


def _episode_sort_key(p: Path) -> tuple[int, str]:
    m = re.search(r"(\d{4})\.json$", p.name)
    return (int(m.group(1)), p.name) if m else (10**9, p.name)


def _pick_d_goal_rows(
    candidate_rows: list[dict[str, Any]],
    *,
    mode: str,
    selected_index: int | None,
) -> list[int | None] | None:
    if not candidate_rows:
        return None
    if mode == "selected":
        if selected_index is None:
            return None
        for row in candidate_rows:
            if int(row.get("candidate_index", -1)) == int(selected_index):
                raw = row.get("d_goal_l2_wm_int")
                if isinstance(raw, list):
                    return [int(x) if x is not None else None for x in raw]
        return None
    if mode == "all_mean":
        lists: list[list[int | None]] = []
        for row in candidate_rows:
            raw = row.get("d_goal_l2_wm_int")
            if isinstance(raw, list) and raw:
                lists.append([int(x) if x is not None else None for x in raw])
        if not lists:
            return None
        width = max(len(lst) for lst in lists)
        out: list[int | None] = []
        for k in range(width):
            vals: list[int] = []
            for lst in lists:
                if k < len(lst) and lst[k] is not None:
                    vals.append(int(lst[k]))  # type: ignore[arg-type]
            if not vals:
                out.append(None)
            else:
                out.append(int(round(float(sum(vals)) / len(vals))))
        return out
    raise ValueError(f"Unknown candidate mode: {mode!r}")


def _segment_bin_values(
    segment: dict[str, Any],
    *,
    candidate_mode: str,
) -> dict[BinKey, float] | None:
    meta = segment.get("metadata") or {}
    rows = meta.get("candidate_wm_goal_l2_int")
    if not isinstance(rows, list) or not rows:
        return None
    stride = meta.get("comparison_wm_env_steps_per_wm_step")
    if stride is None:
        cand0 = (segment.get("candidates") or [{}])[0]
        stride = (cand0.get("meta") or {}).get("wm_env_steps_per_wm_step")
    if stride is None:
        return None
    stride = int(stride)
    start_step = int(segment.get("start_step", 0))
    carried = int(segment.get("carried_steps", 0))
    if carried <= 0:
        c0 = meta.get("comparison_env_step_end")
        c1 = meta.get("comparison_env_step_start")
        if c0 is not None and c1 is not None:
            carried = int(c0) - int(c1)
    if carried <= 0 or stride <= 0:
        return None
    selected_index = segment.get("selected_index")
    if selected_index is not None:
        selected_index = int(selected_index)
    d_list = _pick_d_goal_rows(rows, mode=candidate_mode, selected_index=selected_index)
    if not d_list:
        return None
    out: dict[BinKey, float] = {}
    for k, raw_v in enumerate(d_list):
        if raw_v is None:
            continue
        lo_rel = k * stride
        hi_rel = min((k + 1) * stride, carried)
        if lo_rel >= carried:
            break
        lo_g = start_step + lo_rel
        hi_g = start_step + hi_rel
        out[BinKey(lo_g, hi_g)] = float(raw_v)
    return out or None


def _episode_bin_map(
    payload: dict[str, Any],
    *,
    candidate_mode: str,
) -> dict[BinKey, float] | None:
    """One value per global bin per episode (mean within episode if multiple segments touch same bin)."""
    merged: dict[BinKey, list[float]] = defaultdict(list)
    for seg in payload.get("segments") or []:
        if not isinstance(seg, dict):
            continue
        part = _segment_bin_values(seg, candidate_mode=candidate_mode)
        if not part:
            continue
        for bk, v in part.items():
            merged[bk].append(v)
    if not merged:
        return None
    return {bk: float(sum(vs) / len(vs)) for bk, vs in merged.items()}


def _aggregate_bins(episode_maps: list[dict[BinKey, float]]) -> dict[BinKey, dict[str, float]]:
    by_bin: dict[BinKey, list[float]] = defaultdict(list)
    for ep in episode_maps:
        for bk, v in ep.items():
            by_bin[bk].append(v)
    stats_out: dict[BinKey, dict[str, float]] = {}
    for bk in sorted(by_bin.keys(), key=lambda x: (x.lo, x.hi)):
        vals = by_bin[bk]
        n = len(vals)
        mean_v = sum(vals) / n if n else float("nan")
        if n > 1:
            se = statistics.pstdev(vals) / math.sqrt(n)
        else:
            se = 0.0
        stats_out[bk] = {"n": float(n), "mean": float(mean_v), "se": float(se)}
    return stats_out


def _load_episode_maps(
    run_dir: Path,
    glob_pat: str,
    *,
    candidate_mode: str,
) -> tuple[list[dict[str, Any]], dict[int, dict[BinKey, float]]]:
    paths = sorted(run_dir.glob(glob_pat), key=_episode_sort_key)
    by_episode: dict[int, dict[BinKey, float]] = {}
    errors: list[dict[str, Any]] = []
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(p), "error": repr(exc)})
            continue
        ep_id = data.get("episode_index")
        if ep_id is None:
            errors.append({"path": str(p), "error": "missing episode_index"})
            continue
        ep_id = int(ep_id)
        bmap = _episode_bin_map(data, candidate_mode=candidate_mode)
        if not bmap:
            errors.append({"path": str(p), "error": "no bin map"})
            continue
        by_episode[ep_id] = bmap
    return errors, by_episode


def _stats_to_rows(stats: dict[BinKey, dict[str, float]]) -> list[dict[str, Any]]:
    rows = []
    for bk, st in sorted(stats.items(), key=lambda x: (x[0].lo, x[0].hi)):
        rows.append(
            {
                "range": bk.label(),
                "range_lo": bk.lo,
                "range_hi": bk.hi,
                "n": int(st["n"]),
                "mean": st["mean"],
                "se": st["se"],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], extra_cols: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    if extra_cols:
        for k in extra_cols:
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = dict(r)
            if extra_cols:
                for k, v in extra_cols.items():
                    if k not in row:
                        row[k] = v
            w.writerow(row)


def _latex_box_intro_tabular_note(*, inner_body: str) -> str:
    """Single non-floating block: intro, then tabular (caller supplies), then note; stays in order when pasted."""
    return (
        "% =============================================================================\n"
        "% Do not use em dashes in running text; in LaTeX source use one ASCII hyphen (-), not three in a row.\n"
        "% Paste as one block: INTRO, TABLE, NOTE (fixed order; not a floating table).\n"
        "% If a long chapter breaks inside here, add to preamble: \\usepackage{needspace}\n"
        "% and immediately before this block: \\Needspace{22\\baselineskip}\n"
        "% Optional: \\usepackage{float}, wrap tabular in table[H] still inside minipage.\n"
        "% =============================================================================\n"
        "\n"
        "\\par\\smallskip\n"
        "\\nopagebreak[4]\n"
        "\\begin{minipage}{\\linewidth}\n"
        "\\raggedright\n"
        "\n"
        f"{inner_body}"
        "\n"
        "\\end{minipage}\n"
        "\\par\\smallskip\n"
        "\\nopagebreak[4]\n"
    )


def _emit_latex_solo(
    *,
    intro_lines: list[str],
    rows: list[dict[str, Any]],
    note_lines: list[str],
) -> str:
    body: list[str] = ["% Section: INTRO\n", "\n\n".join(intro_lines), "\n\n% Section: TABLE\n"]
    body.append("\\begin{center}\n")
    body.append("\\begin{tabular}{lrrr}\n\\hline\n")
    body.append("Range & Mean & SE & $n$ \\\\\n\\hline\n")
    for r in rows:
        body.append(f"{r['range']} & {r['mean']:7.2f} & {r['se']:5.2f} & {int(r['n'])} \\\\\n")
    body.append("\\hline\n\\end{tabular}\n")
    body.append("\\end{center}\n\n")
    body.append("% Section: NOTE\n")
    body.append("\n".join(note_lines))
    body.append("\n")
    return _latex_box_intro_tabular_note(inner_body="".join(body))


def _emit_latex_paired(
    *,
    intro_lines: list[str],
    merged_rows: list[dict[str, Any]],
    note_lines: list[str],
    include_delta: bool,
) -> str:
    parts: list[str] = ["% Section: INTRO\n", "\n\n".join(intro_lines), "\n\n% Section: PAIRED TABLE\n"]
    parts.append("\\begin{center}\n")
    if include_delta:
        parts.append("\\begin{tabular}{lrrrrrr}\n\\hline\n")
        parts.append(
            "Range & $\\mu_{\\mathrm{pol}}$ & SE$_{\\mathrm{pol}}$ & $n$ & "
            "$\\mu_{\\mathrm{orc}}$ & SE$_{\\mathrm{orc}}$ & $\\Delta$ \\\\\n\\hline\n"
        )
        for r in merged_rows:
            parts.append(
                f"{r['range']} & {r['mean_policy']:7.2f} & {r['se_policy']:5.2f} & {int(r['n_policy'])} & "
                f"{r['mean_oracle']:7.2f} & {r['se_oracle']:5.2f} & {r['delta_mean']:7.2f} \\\\\n"
            )
    else:
        parts.append("\\begin{tabular}{lrrrrrr}\n\\hline\n")
        parts.append(
            "Range & $\\mu_{\\mathrm{pol}}$ & SE$_{\\mathrm{pol}}$ & $n_{\\mathrm{pol}}$ & "
            "$\\mu_{\\mathrm{orc}}$ & SE$_{\\mathrm{orc}}$ & $n_{\\mathrm{orc}}$ \\\\\n\\hline\n"
        )
        for r in merged_rows:
            parts.append(
                f"{r['range']} & {r['mean_policy']:7.2f} & {r['se_policy']:5.2f} & {int(r['n_policy'])} & "
                f"{r['mean_oracle']:7.2f} & {r['se_oracle']:5.2f} & {int(r['n_oracle'])} \\\\\n"
            )
    parts.append("\\hline\n\\end{tabular}\n")
    parts.append("\\end{center}\n\n")
    parts.append("% Section: NOTE\n")
    parts.append("\n".join(note_lines))
    parts.append("\n")
    return _latex_box_intro_tabular_note(inner_body="".join(parts))


def _merge_paired(
    policy_stats: dict[BinKey, dict[str, float]],
    oracle_stats: dict[BinKey, dict[str, float]],
) -> list[dict[str, Any]]:
    keys = sorted(set(policy_stats.keys()) | set(oracle_stats.keys()), key=lambda x: (x.lo, x.hi))
    rows: list[dict[str, Any]] = []
    for bk in keys:
        ps = policy_stats.get(bk, {"n": 0.0, "mean": float("nan"), "se": 0.0})
        os_ = oracle_stats.get(bk, {"n": 0.0, "mean": float("nan"), "se": 0.0})
        mean_p, se_p, n_p = ps["mean"], ps["se"], int(ps["n"])
        mean_o, se_o, n_o = os_["mean"], os_["se"], int(os_["n"])
        delta = mean_p - mean_o if math.isfinite(mean_p) and math.isfinite(mean_o) else float("nan")
        rows.append(
            {
                "range": bk.label(),
                "range_lo": bk.lo,
                "range_hi": bk.hi,
                "mean_policy": mean_p,
                "se_policy": se_p,
                "n_policy": n_p,
                "mean_oracle": mean_o,
                "se_oracle": se_o,
                "n_oracle": n_o,
                "delta_mean": delta,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True, help="Primary run directory (phase08 policy).")
    p.add_argument("--glob", type=str, default="", help="Episode JSON glob (default: infer).")
    p.add_argument(
        "--compare-run-dir",
        type=Path,
        default=None,
        help="Optional second run (e.g. phase09 oracle replay) for paired export.",
    )
    p.add_argument("--compare-glob", type=str, default="", help="Glob for compare run (default: infer).")
    p.add_argument("--candidate-mode", choices=["selected", "all_mean"], default="selected")
    p.add_argument("--out-prefix", type=str, default="latent_range_baseline")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: primary --run-dir).")
    p.add_argument("--latex-out", type=Path, default=None, help="Write LaTeX fragment here.")
    p.add_argument(
        "--latex-include-delta",
        action="store_true",
        help="Include Delta column in paired LaTeX (default: off).",
    )
    args = p.parse_args(argv)

    run_dir = args.run_dir.expanduser().resolve()
    out_dir = (args.out_dir or run_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    glob_primary = args.glob.strip() or _infer_glob(run_dir)

    err_p, by_ep_p = _load_episode_maps(run_dir, glob_primary, candidate_mode=args.candidate_mode)
    maps_p = list(by_ep_p.values())
    stats_p = _aggregate_bins(maps_p)
    rows_p = _stats_to_rows(stats_p)

    prefix = args.out_prefix.strip() or "latent_range_baseline"
    json_primary = {
        "run_dir": str(run_dir),
        "glob": glob_primary,
        "candidate_mode": args.candidate_mode,
        "n_episodes": len(by_ep_p),
        "parse_errors": err_p,
        "bins": rows_p,
    }

    manifest = run_dir / "segment_grpo_manifest.json"
    if manifest.is_file():
        try:
            json_primary["manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            json_primary["manifest_path"] = str(manifest)

    (out_dir / f"{prefix}_policy.json").write_text(
        json.dumps(json_primary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if rows_p:
        _write_csv(out_dir / f"{prefix}_policy.csv", rows_p)

    intro = [
        r"\noindent\textbf{Setup (policy run).} Meta-World \texttt{push-v3}; WM-goal distance is Euclidean L2 in \texttt{visual} scoring latent, integer-rounded per WM megastep (artifact \texttt{d\_goal\_l2\_wm\_int}).",
        f"Run directory: \\texttt{{{_latex_escape(str(run_dir))}}}; episodes aggregated: {len(by_ep_p)}; candidate mode: \\texttt{{{args.candidate_mode}}}.",
    ]

    note = [
        r"\medskip\noindent\footnotesize\textit{Note.} Table shows cross-episode mean $\pm$ SE of per-bin values; bins are half-open env action step ranges aligned to \texttt{comparison\_wm\_env\_steps\_per\_wm\_step}. Not pixel MSE to goal image.",
    ]

    if args.compare_run_dir is None:
        if args.latex_out and rows_p:
            tex = _emit_latex_solo(intro_lines=intro, rows=rows_p, note_lines=note)
            args.latex_out.parent.mkdir(parents=True, exist_ok=True)
            args.latex_out.write_text(tex, encoding="utf-8")
        print(f"[aggregate] policy episodes={len(by_ep_p)} bins={len(rows_p)} out_dir={out_dir}")
        return 0

    run_o = args.compare_run_dir.expanduser().resolve()
    glob_o = args.compare_glob.strip() or _infer_glob(run_o)
    err_o, by_ep_o = _load_episode_maps(run_o, glob_o, candidate_mode="selected")
    # Reorder policy maps by same episode keys present in both
    common = sorted(set(by_ep_p.keys()) & set(by_ep_o.keys()))
    maps_p_aligned = [by_ep_p[i] for i in common]
    maps_o_aligned = [by_ep_o[i] for i in common]
    stats_p2 = _aggregate_bins(maps_p_aligned)
    stats_o = _aggregate_bins(maps_o_aligned)
    merged = _merge_paired(stats_p2, stats_o)

    json_pair = {
        "policy_run_dir": str(run_dir),
        "oracle_replay_run_dir": str(run_o),
        "policy_glob": glob_primary,
        "oracle_glob": glob_o,
        "candidate_mode_policy": args.candidate_mode,
        "n_episodes_policy": len(by_ep_p),
        "n_episodes_oracle": len(by_ep_o),
        "n_episodes_paired": len(common),
        "parse_errors_policy": err_p,
        "parse_errors_oracle": err_o,
        "paired_bins": merged,
    }
    (out_dir / f"{prefix}_paired.json").write_text(
        json.dumps(json_pair, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if merged:
        _write_csv(out_dir / f"{prefix}_paired.csv", merged)

    intro_paired = intro + [
        r"\noindent\textbf{Setup (oracle replay run).} Same WM and goal latent definition; actions come from recorded oracle \texttt{actions.jsonl} (phase09-style replay).",
        f"Compare directory: \\texttt{{{_latex_escape(str(run_o))}}}; paired episodes: {len(common)}.",
    ]

    note_paired = [
        r"\medskip",
        r"\noindent\footnotesize\textit{Note.} Table shows cross-episode mean $\pm$ SE of per-bin values; "
        r"bins are half-open env action step ranges aligned to \texttt{comparison\_wm\_env\_steps\_per\_wm\_step}. "
        r"Not pixel MSE to goal image.",
        r"\noindent\footnotesize\textit{Oracle column:} WM conditioned on expert action sequence, "
        r"not a second world model.",
        r"\noindent\footnotesize\textit{Policy column:} SmolVLA-selected chunk ($K{=}3$ run) per artifact defaults.",
    ]

    if args.latex_out and merged:
        tex = _emit_latex_paired(
            intro_lines=intro_paired,
            merged_rows=merged,
            note_lines=note_paired,
            include_delta=bool(args.latex_include_delta),
        )
        args.latex_out.parent.mkdir(parents=True, exist_ok=True)
        args.latex_out.write_text(tex, encoding="utf-8")

    print(
        f"[aggregate] policy_ep={len(by_ep_p)} oracle_ep={len(by_ep_o)} paired={len(common)} "
        f"bins={len(merged)} out_dir={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
