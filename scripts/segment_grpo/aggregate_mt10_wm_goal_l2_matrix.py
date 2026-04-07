#!/usr/bin/env python3
"""Cross-task MT10 matrix: mean WM-goal latent L2 per env action bin (phase08 vs phase09).

Discovers one ``mt10_run_*`` directory per task under each baseline root (via
``segment_grpo_manifest.json`` field ``task``), reuses bin aggregation from
``aggregate_wm_goal_l2_by_action_range.py``, emits wide JSON/CSV/LaTeX.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SEG_SCRIPTS = ROOT / "scripts" / "segment_grpo"


def _load_agg_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "aggregate_wm_goal_l2_by_action_range",
        SEG_SCRIPTS / "aggregate_wm_goal_l2_by_action_range.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["aggregate_wm_goal_l2_by_action_range"] = mod
    spec.loader.exec_module(mod)
    return mod


def _discover_task_run_dirs(artifacts_root: Path) -> dict[str, Path]:
    """Map task name -> run directory (parent of segment_grpo_manifest.json)."""
    root = artifacts_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    out: dict[str, Path] = {}
    for manifest in sorted(root.glob("mt10_run_*/segment_grpo_manifest.json")):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        task = data.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError(f"Missing or invalid task in {manifest}")
        task = task.strip()
        run_dir = manifest.parent
        if task in out:
            raise ValueError(f"Duplicate task {task!r} under {root}:\n  {out[task]}\n  {run_dir}")
        out[task] = run_dir
    return out


def _expected_bin_labels(*, max_env_steps: int, stride: int = 5) -> list[str]:
    if max_env_steps < 1:
        raise ValueError(f"max_env_steps must be >= 1, got {max_env_steps}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    return [f"{lo}:{min(lo + stride, max_env_steps)}" for lo in range(0, max_env_steps, stride)]


# Row labels in CSV/LaTeX second column (phase08 policy vs phase09 oracle replay).
PHASE8_ROW_LABEL = "VLA"
PHASE9_ROW_LABEL = "oracle"
OVERALL_TASK_LABEL = "All tasks"


def _csv_col_for_range(label: str) -> str:
    return "r" + label.replace(":", "_")


def _goal_bin_index(bin_labels: list[str], goal_frame_index: int | None) -> int | None:
    if goal_frame_index is None:
        return None
    for i, lab in enumerate(bin_labels):
        try:
            lo, hi = map(int, lab.split(":"))
        except ValueError:
            continue
        if lo < goal_frame_index <= hi:
            return i
    return None


def _tabular_cols_with_goal_sep(prefix: str, n: int, goal_bin_index: int | None) -> str:
    cols = [*prefix]
    if n <= 0:
        return "".join(cols)
    for i in range(n):
        if goal_bin_index is not None and i == goal_bin_index:
            cols.append("|")
        cols.append("r")
        if goal_bin_index is not None and i == goal_bin_index:
            cols.append("|")
    return "".join(cols)


def _aggregate_one_run(
    agg: Any,
    run_dir: Path,
    *,
    candidate_mode: str,
    max_env_steps: int,
) -> dict[str, Any]:
    glob_pat = agg._infer_glob(run_dir)
    err, by_ep = agg._load_episode_maps(run_dir, glob_pat, candidate_mode=candidate_mode)
    maps = list(by_ep.values())
    stats = agg._aggregate_bins(maps) if maps else {}
    rows = agg._stats_to_rows(stats)
    by_label = {
        r["range"]: r
        for r in rows
        if int(r.get("range_hi", 0)) <= int(max_env_steps)
    }
    return {
        "run_dir": str(run_dir),
        "glob": glob_pat,
        "candidate_mode": candidate_mode,
        "n_episodes": len(by_ep),
        "parse_errors": err,
        "stats_by_range": by_label,
        "stats_rows": rows,
    }


def _wide_row(
    task: str,
    phase: str,
    by_label: dict[str, dict[str, Any]],
    bin_labels: list[str],
) -> dict[str, Any]:
    row: dict[str, Any] = {"task": task, "phase": phase}
    for lab in bin_labels:
        col = _csv_col_for_range(lab)
        cell = by_label.get(lab)
        if cell is None:
            row[col] = ""
        elif math.isfinite(float(cell["mean"])):
            row[col] = float(cell["mean"])
        else:
            row[col] = ""
    return row


def _normalize_task_name(task: str) -> str:
    return task[:-3] if task.endswith("-v3") else task


def _format_latex_number(v: Any, *, signed: bool = False) -> str:
    if v == "" or v is None:
        return "---"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)) or (
        isinstance(v, str) and _looks_like_number(v)
    ):
        f = float(v)
        if math.isfinite(f):
            return f"{f:+.2f}" if signed else f"{f:.2f}"
        return "---"
    return str(v)


def _looks_like_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _round_int(v: float | None) -> int | str:
    if v is None:
        return ""
    return int(round(v))


def _overall_rows(
    per_task: dict[str, dict[str, Any]],
    bin_labels: list[str],
) -> dict[str, Any]:
    """Average VLA/oracle values across tasks (macro-averaged by task means)."""
    out: dict[str, Any] = {}
    vla_bins: dict[str, list[float]] = {lab: [] for lab in bin_labels}
    oracle_bins: dict[str, list[float]] = {lab: [] for lab in bin_labels}

    for task_payload in per_task.values():
        for lab in bin_labels:
            vla_cell = task_payload["phase8"]["stats_by_range"].get(lab)
            oracle_cell = task_payload["phase9"]["stats_by_range"].get(lab)
            if vla_cell is not None and math.isfinite(float(vla_cell.get("mean"))):
                vla_bins[lab].append(float(vla_cell["mean"]))
            if oracle_cell is not None and math.isfinite(float(oracle_cell.get("mean"))):
                oracle_bins[lab].append(float(oracle_cell["mean"]))

    for lab in bin_labels:
        vla_vals = vla_bins[lab]
        oracle_vals = oracle_bins[lab]
        vla_mean = sum(vla_vals) / len(vla_vals) if vla_vals else None
        oracle_mean = sum(oracle_vals) / len(oracle_vals) if oracle_vals else None
        out[lab] = {
            "vla_mean": vla_mean,
            "oracle_mean": oracle_mean,
            "delta": None if vla_mean is None or oracle_mean is None else oracle_mean - vla_mean,
            "n_tasks_vla": len(vla_vals),
            "n_tasks_oracle": len(oracle_vals),
        }
    return out


def _summary_value(values: list[float] | list[int]) -> float | None:
    if not values:
        return None
    return sum(float(v) for v in values) / len(values)


def _summary_int(values: list[float] | list[int]) -> int | str:
    v = _summary_value(values)
    if v is None:
        return ""
    return int(round(v))


def _compare_col_for_range(label: str, suffix: str) -> str:
    return f"r{label.replace(':', '_')}_{suffix}"


def _compare_rows(
    per_task: dict[str, dict[str, Any]],
    tasks_sorted: list[str],
    bin_labels: list[str],
    overall: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks_sorted:
        task_payload = per_task[task]
        vla_by_label = task_payload["phase8"]["stats_by_range"]
        oracle_by_label = task_payload["phase9"]["stats_by_range"]
        row: dict[str, Any] = {"task": task}
        for lab in bin_labels:
            vla_cell = vla_by_label.get(lab)
            oracle_cell = oracle_by_label.get(lab)
            vla_v = int(round(float(vla_cell["mean"]))) if vla_cell and math.isfinite(float(vla_cell["mean"])) else ""
            orc_v = int(round(float(oracle_cell["mean"]))) if oracle_cell and math.isfinite(float(oracle_cell["mean"])) else ""
            delta_v = int(round(float(orc_v) - float(vla_v))) if isinstance(vla_v, int) and isinstance(orc_v, int) else ""
            row[_compare_col_for_range(lab, "VLA")] = vla_v
            row[_compare_col_for_range(lab, "oracle")] = orc_v
            row[_compare_col_for_range(lab, "delta")] = delta_v
        rows.append(row)

    overall_row: dict[str, Any] = {"task": OVERALL_TASK_LABEL}
    for lab in bin_labels:
        o = overall[lab]
        vla_v = _round_int(o["vla_mean"])
        orc_v = _round_int(o["oracle_mean"])
        delta = _round_int(o["delta"]) if o["delta"] is not None else ""
        overall_row[_compare_col_for_range(lab, "VLA")] = vla_v
        overall_row[_compare_col_for_range(lab, "oracle")] = orc_v
        overall_row[_compare_col_for_range(lab, "delta")] = delta
    rows.append(overall_row)
    return rows


def _compare_rows_long(
    per_task: dict[str, dict[str, Any]],
    tasks_sorted: list[str],
    bin_labels: list[str],
    overall: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks_sorted:
        task_payload = per_task[task]
        vla_by_label = task_payload["phase8"]["stats_by_range"]
        oracle_by_label = task_payload["phase9"]["stats_by_range"]
        for lab in bin_labels:
            vla_cell = vla_by_label.get(lab)
            oracle_cell = oracle_by_label.get(lab)
            vla_v = int(round(float(vla_cell["mean"]))) if vla_cell and math.isfinite(float(vla_cell["mean"])) else ""
            orc_v = int(round(float(oracle_cell["mean"]))) if oracle_cell and math.isfinite(float(oracle_cell["mean"])) else ""
            delta_v = int(round(float(orc_v) - float(vla_v))) if isinstance(vla_v, int) and isinstance(orc_v, int) else ""
            rows.append(
                {
                    "task": task,
                    "range": lab,
                    "vla": vla_v,
                    "oracle": orc_v,
                    "delta": delta_v,
                }
            )

    for lab in bin_labels:
        o = overall[lab]
        vla_v = _round_int(o["vla_mean"])
        orc_v = _round_int(o["oracle_mean"])
        delta = _round_int(o["delta"]) if o["delta"] is not None else ""
        rows.append(
            {
                "task": OVERALL_TASK_LABEL,
                "range": lab,
                "vla": vla_v,
                "oracle": orc_v,
                "delta": delta,
            }
        )
    return rows


def _compare_rows_summary(
    per_task: dict[str, dict[str, Any]],
    tasks_sorted: list[str],
    bin_labels: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks_sorted:
        task_payload = per_task[task]
        vla_vals: list[float] = []
        oracle_vals: list[float] = []
        for lab in bin_labels:
            vla_cell = task_payload["phase8"]["stats_by_range"].get(lab)
            oracle_cell = task_payload["phase9"]["stats_by_range"].get(lab)
            if vla_cell is not None and math.isfinite(float(vla_cell["mean"])):
                vla_vals.append(float(vla_cell["mean"]))
            if oracle_cell is not None and math.isfinite(float(oracle_cell["mean"])):
                oracle_vals.append(float(oracle_cell["mean"]))

        vla_v = _summary_int(vla_vals)
        orc_v = _summary_int(oracle_vals)
        delta = int(round(float(orc_v) - float(vla_v))) if isinstance(vla_v, int) and isinstance(orc_v, int) else ""
        rows.append(
            {
                "task": task,
                "vla": vla_v,
                "oracle": orc_v,
                "delta": delta,
            }
        )

    overall = [row for row in rows if row["task"] != OVERALL_TASK_LABEL]
    all_vla = _summary_int(
        [float(v["vla"]) for v in overall if isinstance(v["vla"], int)]
    )
    all_oracle = _summary_int(
        [float(v["oracle"]) for v in overall if isinstance(v["oracle"], int)]
    )
    all_delta = (
        int(round(float(all_oracle) - float(all_vla)))
        if isinstance(all_vla, int) and isinstance(all_oracle, int)
        else ""
    )
    rows.append({"task": OVERALL_TASK_LABEL, "vla": all_vla, "oracle": all_oracle, "delta": all_delta})
    return rows


def _compare_rows_run_delta(
    per_task: dict[str, dict[str, Any]],
    tasks_sorted: list[str],
    overall: dict[str, Any],
    bin_labels: list[str],
) -> list[dict[str, Any]]:
    # Preserve deterministic task order by iterating over tasks_sorted directly.
    rows: list[dict[str, Any]] = []
    for task in tasks_sorted:
        task_payload = per_task[task]
        vla_by_label = task_payload["phase8"]["stats_by_range"]
        oracle_by_label = task_payload["phase9"]["stats_by_range"]

        task_display = _normalize_task_name(task)
        vla_row: dict[str, Any] = {"task": task_display, "metric": "VLA"}
        delta_row: dict[str, Any] = {"task": task_display, "metric": "Delta (oracle)"}
        for lab in bin_labels:
            vla_cell = vla_by_label.get(lab)
            oracle_cell = oracle_by_label.get(lab)
            vla_v = float(vla_cell["mean"]) if vla_cell and math.isfinite(float(vla_cell["mean"])) else ""
            oracle_v = float(oracle_cell["mean"]) if oracle_cell and math.isfinite(float(oracle_cell["mean"])) else ""
            vla_row[_csv_col_for_range(lab)] = vla_v
            if isinstance(vla_v, float) and isinstance(oracle_v, float):
                delta_row[_csv_col_for_range(lab)] = oracle_v - vla_v
            else:
                delta_row[_csv_col_for_range(lab)] = ""

        rows.append(vla_row)
        rows.append(delta_row)

    overall_row_vla: dict[str, Any] = {"task": OVERALL_TASK_LABEL, "metric": "VLA"}
    overall_row_delta: dict[str, Any] = {
        "task": OVERALL_TASK_LABEL,
        "metric": "Delta (oracle)",
    }
    for lab in bin_labels:
        o = overall[lab]
        o_vla = float(o["vla_mean"]) if o["vla_mean"] is not None else None
        o_oracle = float(o["oracle_mean"]) if o["oracle_mean"] is not None else None
        overall_row_vla[_csv_col_for_range(lab)] = o_vla
        if o_vla is not None and o_oracle is not None:
            overall_row_delta[_csv_col_for_range(lab)] = float(o_oracle) - float(o_vla)
        else:
            overall_row_delta[_csv_col_for_range(lab)] = ""
    rows.append(overall_row_vla)
    rows.append(overall_row_delta)
    return rows


def _emit_latex_compare_wide(
    *,
    bin_labels: list[str],
    compare_rows: list[dict[str, Any]],
    intro_lines: list[str],
) -> str:
    cols = "l" + "r" * (3 * len(bin_labels))
    headers: list[str] = [r"\textbf{Task}"]
    for lab in bin_labels:
        headers.extend(
            [
                rf"\textbf{{{_latex_escape(lab)} VLA}}",
                rf"\textbf{{{_latex_escape(lab)} oracle}}",
                rf"\textbf{{{_latex_escape(lab)} delta}}",
            ]
        )
    head_cells = " & ".join(headers)
    body: list[str] = [
        "% MT10 WM-goal latent L2 paired table: VLA vs oracle per range.\n",
        "% Requires \\texttt{graphicx} for \\texttt{\\textbackslash resizebox}.\n",
        "% Paste fragment; add booktabs to preamble if desired.\n\n",
        "\n".join(intro_lines),
        "\n\n\\begin{center}\n",
        "\\resizebox{\\textwidth}{!}{%\n",
        f"\\begin{{tabular}}{{{cols}}}\n\\hline\n",
        head_cells + " \\\\\n\\hline\n",
    ]
    for row in compare_rows:
        task = _normalize_task_name(str(row["task"]))
        if task == OVERALL_TASK_LABEL:
            body.append("\\hline\n")
        cells = [_latex_escape(task)]
        for lab in bin_labels:
            for suffix in ("VLA", "oracle", "delta"):
                col = _compare_col_for_range(lab, suffix)
                v = row.get(col, "")
                cells.append(_format_latex_number(v))
        body.append(" & ".join(cells) + " \\\\\n")
    body.append("\\hline\n\\end{tabular}\n}\n\\end{center}\n")
    return "".join(body)


def _emit_latex_compare_summary(
    *,
    compare_rows: list[dict[str, Any]],
    intro_lines: list[str],
) -> str:
    cols = "lrrr"
    body: list[str] = [
        "% MT10 WM-goal latent L2 paired summary table.\n",
        "% Requires \\texttt{graphicx} for \\texttt{\\textbackslash resizebox}.\n",
        "% Paste fragment; add booktabs to preamble if desired.\n\n",
        "\n".join(intro_lines),
        "\n\n\\begin{center}\n",
        "\\resizebox{\\textwidth}{!}{%\n",
        f"\\begin{{tabular}}{{{cols}}}\n\\hline\n",
        "\\textbf{Task} & \\textbf{VLA} & \\textbf{oracle} & \\textbf{delta} \\\\\n",
        "\\hline\n",
    ]
    for row in compare_rows:
        task = _latex_escape(_normalize_task_name(str(row.get("task", ""))))
        vla = row.get("vla", "")
        oracle = row.get("oracle", "")
        delta = row.get("delta", "")
        if task == OVERALL_TASK_LABEL:
            body.append("\\hline\n")
        vals = [
            task,
            f"{vla}" if isinstance(vla, int) else vla,
            f"{oracle}" if isinstance(oracle, int) else oracle,
            f"{delta}" if isinstance(delta, int) else delta,
        ]
        body.append(" & ".join(vals) + " \\\\\n")
    body.append("\\hline\n\\end{tabular}\n}\n\\end{center}\n")
    return "".join(body)


def _emit_latex_compare_long(
    *,
    compare_rows: list[dict[str, Any]],
    intro_lines: list[str],
) -> str:
    cols = "llrrr"
    body: list[str] = [
        "% MT10 WM-goal latent L2 paired table: VLA vs oracle per range.\n",
        "% Requires \\texttt{graphicx} for \\texttt{\\textbackslash resizebox}.\n",
        "% Paste fragment; add booktabs to preamble if desired.\n\n",
        "\n".join(intro_lines),
        "\n\n\\begin{center}\n",
        "\\resizebox{\\textwidth}{!}{%\n",
        f"\\begin{{tabular}}{{{cols}}}\n\\hline\n",
        "\\textbf{Task} & \\textbf{Range} & \\textbf{VLA} & \\textbf{oracle} & \\textbf{delta} \\\\\n",
        "\\hline\n",
    ]
    inserted_overall = False
    for row in compare_rows:
        task = _latex_escape(_normalize_task_name(str(row.get("task", ""))))
        rng = _latex_escape(str(row.get("range", "")))
        vla = row.get("vla", "")
        oracle = row.get("oracle", "")
        delta = row.get("delta", "")
        if task == OVERALL_TASK_LABEL and not inserted_overall:
            body.append("\\hline\n")
            inserted_overall = True
        vals = [
            task,
            rng,
            f"{vla}" if isinstance(vla, int) else vla,
            f"{oracle}" if isinstance(oracle, int) else oracle,
            f"{delta}" if isinstance(delta, int) else delta,
        ]
        body.append(" & ".join(vals) + " \\\\\n")
    body.append("\\hline\n\\end{tabular}\n}\n\\end{center}\n")
    return "".join(body)


def _emit_latex_compare_run_delta(
    *,
    bin_labels: list[str],
    compare_rows: list[dict[str, Any]],
    intro_lines: list[str],
    goal_frame_index: int | None = None,
) -> str:
    goal_bin = _goal_bin_index(bin_labels, goal_frame_index)
    cols = _tabular_cols_with_goal_sep("ll", len(bin_labels), goal_bin)
    head_cells = " & ".join([r"\textbf{Task}", r"\textbf{Run}"] + [rf"\textbf{{{_latex_escape(lab)}}}" for lab in bin_labels])
    body: list[str] = [
        "% MT10 WM-goal latent L2 task-wise table: VLA and oracle delta per range.\n",
        "% Requires \\texttt{graphicx} for \\texttt{\\textbackslash resizebox}.\n",
        "% Paste fragment; add booktabs to preamble if desired.\n\n",
        "\n".join(intro_lines),
        "\n\n\\begin{center}\n",
        "\\resizebox{\\textwidth}{!}{%\n",
        f"\\begin{{tabular}}{{{cols}}}\n\\hline\n",
        head_cells + " \\\\\n\\hline\n",
    ]
    prev_task: str | None = None
    for row in compare_rows:
        task = _normalize_task_name(str(row.get("task", "")))
        metric = str(row.get("metric", ""))
        metric_display = r"$\Delta_{\mathrm{oracle}}$" if metric.startswith("Delta") else metric
        values = [_latex_escape(task), metric_display]
        is_delta = metric.startswith("Delta")
        if prev_task is not None and task != prev_task:
            body.append("\\hline\n")
        for lab in bin_labels:
            v = row.get(_csv_col_for_range(lab), "")
            values.append(_format_latex_number(v, signed=is_delta))
        body.append(" & ".join(values) + " \\\\\n")
        if task == OVERALL_TASK_LABEL and metric == "Delta (oracle)":
            body.append("\\hline\n")
        prev_task = task
    body.append("\\hline\n\\end{tabular}\n}\n\\end{center}\n")
    body.append(
        "\\par\\vspace{0.3em}\n"
        "\\noindent\\small\\textbf{How to read this table.} "
        "Cell values are cross-episode mean latent distances per bin. "
        "Each task has a VLA row and a \\(\\Delta_{\\mathrm{oracle}}\\) row, where "
        "\\(\\Delta_{\\mathrm{oracle}} = \\mathrm{oracle} - \\mathrm{VLA}\\). "
        "A positive value means oracle is farther from the world-model goal than VLA, "
        "and a negative value means oracle is closer. "
        "In this representation, larger raw values mean larger latent distance, "
        "so values closer to zero are preferable when the distance is interpreted as reconstruction error.\n"
    )
    return "".join(body)


def _latex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def _emit_latex(
    *,
    bin_labels: list[str],
    wide_rows: list[dict[str, Any]],
    intro_lines: list[str],
) -> str:
    """Wide tabular: Task | Phase | ten range columns (means only)."""
    cols = "ll" + "r" * len(bin_labels)
    head_cells = " & ".join([r"\textbf{Task}", r"\textbf{Run}"] + [rf"\textbf{{{_latex_escape(lab)}}}" for lab in bin_labels])
    body: list[str] = [
        "% MT10 WM-goal latent L2 per 5-step bin, cross-episode mean per bin.\n",
        "% Requires \\texttt{graphicx} for \\texttt{\\textbackslash resizebox}.\n",
        "% Paste fragment; add booktabs to preamble if desired.\n\n",
        "\n".join(intro_lines),
        "\n\n\\begin{center}\n",
        "\\resizebox{\\textwidth}{!}{%\n",
        f"\\begin{{tabular}}{{{cols}}}\n\\hline\n",
        head_cells + " \\\\\n\\hline\n",
    ]
    for wr in wide_rows:
        task = _normalize_task_name(str(wr["task"]))
        metric = wr["phase"]
        metric_label = r"$\Delta_{\mathrm{oracle}}$" if metric.startswith("Delta") else metric
        cells = [_latex_escape(task), metric_label]
        for lab in bin_labels:
            col = _csv_col_for_range(lab)
            v = wr.get(col, "")
            if v == "" or v is None:
                cells.append("---")
            else:
                cells.append(_format_latex_number(v, signed=metric.startswith("Delta")))
        body.append(" & ".join(cells) + " \\\\\n")
    body.append("\\hline\n\\end{tabular}\n}\n\\end{center}\n")
    return "".join(body)


def build_matrix_payload(
    *,
    phase8_root: Path,
    phase9_root: Path,
    candidate_mode_phase8: str,
    candidate_mode_phase9: str,
    max_env_steps: int,
    strict: bool,
) -> dict[str, Any]:
    agg = _load_agg_module()
    if candidate_mode_phase8 not in ("selected", "all_mean"):
        raise ValueError(f"Invalid candidate_mode_phase8: {candidate_mode_phase8!r}")
    if candidate_mode_phase9 not in ("selected", "all_mean"):
        raise ValueError(f"Invalid candidate_mode_phase9: {candidate_mode_phase9!r}")

    p8 = _discover_task_run_dirs(phase8_root)
    p9 = _discover_task_run_dirs(phase9_root)
    keys8, keys9 = set(p8), set(p9)
    if keys8 != keys9:
        only8 = sorted(keys8 - keys9)
        only9 = sorted(keys9 - keys8)
        msg = f"Task mismatch between phase8 and phase9.\n  Only phase8: {only8}\n  Only phase9: {only9}"
        if strict:
            raise ValueError(msg)
    common = sorted(keys8 & keys9)
    if strict and len(common) != 10:
        raise ValueError(f"Expected 10 paired tasks, got {len(common)}: {common}")

    bin_labels = _expected_bin_labels(max_env_steps=max_env_steps, stride=5)
    goal_frame_index: int | None = None
    for task in common:
        m8 = json.loads((p8[task] / "segment_grpo_manifest.json").read_text(encoding="utf-8"))
        m9 = json.loads((p9[task] / "segment_grpo_manifest.json").read_text(encoding="utf-8"))
        g8 = m8.get("goal_frame_index")
        g9 = m9.get("goal_frame_index")
        if not isinstance(g8, int) or not isinstance(g9, int):
            raise ValueError(f"Missing/invalid goal_frame_index for task={task}")
        if g8 != g9:
            raise ValueError(
                f"Goal-frame mismatch for task={task}: phase8={g8}, phase9={g9}. "
                "Re-run phase9 with MT10_PHASE9_GOAL_FRAME=25 and matching root."
            )
        if goal_frame_index is None:
            goal_frame_index = g8
        elif goal_frame_index != g8:
            raise ValueError(
                f"Inconsistent goal_frame_index for task={task}: expected {goal_frame_index}, got {g8}."
            )
    per_task: dict[str, Any] = {}
    wide_rows: list[dict[str, Any]] = []

    for task in common:
        s8 = _aggregate_one_run(
            agg,
            p8[task],
            candidate_mode=candidate_mode_phase8,
            max_env_steps=max_env_steps,
        )
        s9 = _aggregate_one_run(
            agg,
            p9[task],
            candidate_mode=candidate_mode_phase9,
            max_env_steps=max_env_steps,
        )
        per_task[task] = {"phase8": s8, "phase9": s9}
        wide_rows.append(_wide_row(task, PHASE8_ROW_LABEL, s8["stats_by_range"], bin_labels))
        wide_rows.append(_wide_row(task, PHASE9_ROW_LABEL, s9["stats_by_range"], bin_labels))

    overall = _overall_rows(per_task, bin_labels)
    compare_rows = _compare_rows(per_task, common, bin_labels, overall)
    compare_rows_long = _compare_rows_long(per_task, common, bin_labels, overall)
    compare_rows_summary = _compare_rows_summary(per_task, common, bin_labels)
    # compare_rows_run_delta keeps `common` task order and then appends All-tasks rows.
    compare_rows_run_delta = _compare_rows_run_delta(per_task, common, overall, bin_labels)

    return {
        "phase8_root": str(phase8_root.expanduser().resolve()),
        "phase9_root": str(phase9_root.expanduser().resolve()),
        "candidate_mode_phase8": candidate_mode_phase8,
        "candidate_mode_phase9": candidate_mode_phase9,
        "bin_labels": bin_labels,
        "tasks_sorted": common,
        "n_tasks": len(common),
        "per_task": per_task,
        "wide_csv_rows": wide_rows,
        "overall_task_avgs": overall,
        "compare_csv_rows": compare_rows,
        "compare_csv_rows_long": compare_rows_long,
        "compare_csv_rows_summary": compare_rows_summary,
        "compare_csv_rows_run_delta": compare_rows_run_delta,
        "max_env_steps": int(max_env_steps),
        "goal_frame_index": goal_frame_index,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            fieldnames = ["task"]
        else:
            fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--phase8-root",
        type=Path,
        required=True,
        help="Parent of mt10_run_* phase08 (policy) directories.",
    )
    p.add_argument(
        "--phase9-root",
        type=Path,
        required=True,
        help="Parent of mt10_run_* phase09 (oracle replay) directories.",
    )
    p.add_argument(
        "--candidate-mode-phase8",
        choices=["selected", "all_mean"],
        default="selected",
        help="Candidate handling for phase08 (K>1 runs).",
    )
    p.add_argument(
        "--candidate-mode-phase9",
        choices=["selected", "all_mean"],
        default="selected",
        help="Candidate handling for phase09 (usually single candidate).",
    )
    p.add_argument("--out-json", type=Path, default=None, help="Write full payload JSON.")
    p.add_argument("--out-csv", type=Path, default=None, help="Write wide 20-row CSV.")
    p.add_argument("--latex-out", type=Path, default=None, help="Write LaTeX tabular fragment.")
    p.add_argument(
        "--out-csv-compare",
        type=Path,
        default=None,
        help="Write compare CSV by compare format (includes all-tasks average rows when applicable).",
    )
    p.add_argument(
        "--out-latex-compare",
        type=Path,
        default=None,
        help="Write compare LaTeX table (format selected with --compare-format).",
    )
    p.add_argument(
        "--compare-format",
        choices=["wide", "long", "summary", "run-delta"],
        default="run-delta",
        help="Compare output shape: run-delta (task+metric rows), wide(10*3 cols), long(task+range rows), summary(11 rows).",
    )
    p.add_argument(
        "--max-env-steps",
        type=int,
        default=50,
        help="Maximum env-action steps per row/bin to include (default: 50).",
    )
    p.add_argument(
        "--no-strict",
        action="store_true",
        help="Allow task-set mismatch between roots (still errors on duplicate task).",
    )
    args = p.parse_args(argv)

    payload = build_matrix_payload(
        phase8_root=args.phase8_root,
        phase9_root=args.phase9_root,
        candidate_mode_phase8=args.candidate_mode_phase8,
        candidate_mode_phase9=args.candidate_mode_phase9,
        max_env_steps=int(args.max_env_steps),
        strict=not args.no_strict,
    )
    bin_labels = payload["bin_labels"]
    wide_rows = payload["wide_csv_rows"]

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.out_csv:
        _write_csv(
            args.out_csv,
            wide_rows,
            ["task", "phase"] + [_csv_col_for_range(lab) for lab in bin_labels],
        )
    if args.out_csv_compare:
        if args.compare_format == "wide":
            cmp_fieldnames: list[str] = ["task"]
            for lab in bin_labels:
                cmp_fieldnames.extend(
                    [
                        _compare_col_for_range(lab, "VLA"),
                        _compare_col_for_range(lab, "oracle"),
                        _compare_col_for_range(lab, "delta"),
                    ]
                )
            _write_csv(args.out_csv_compare, payload["compare_csv_rows"], cmp_fieldnames)
        elif args.compare_format == "run-delta":
            _write_csv(
                args.out_csv_compare,
                payload["compare_csv_rows_run_delta"],
                ["task", "metric"] + [_csv_col_for_range(lab) for lab in bin_labels],
            )
        elif args.compare_format == "summary":
            _write_csv(
                args.out_csv_compare,
                payload["compare_csv_rows_summary"],
                ["task", "vla", "oracle", "delta"],
            )
        else:
            _write_csv(
                args.out_csv_compare,
                payload["compare_csv_rows_long"],
                ["task", "range", "vla", "oracle", "delta"],
            )
    if args.latex_out:
        intro = [
            r"\noindent\textbf{MT10 WM-goal latent L2.} Cross-episode mean per env step bin; "
            r"integer L2 in WM visual latent (artifact \texttt{d\_goal\_l2\_wm\_int}). "
            r"\textbf{VLA:} phase08 SmolVLA policy; \textbf{oracle:} phase09 oracle-action replay.",
            f"Phase8 root: \\texttt{{{_latex_escape(payload['phase8_root'])}}}; "
            f"Phase9 root: \\texttt{{{_latex_escape(payload['phase9_root'])}}}; "
            f"analysis max env steps: \\texttt{{{payload['max_env_steps']}}}.",
        ]
        tex = _emit_latex(bin_labels=bin_labels, wide_rows=wide_rows, intro_lines=intro)
        args.latex_out.parent.mkdir(parents=True, exist_ok=True)
        args.latex_out.write_text(tex, encoding="utf-8")
    if args.out_latex_compare:
        intro = [
            r"\noindent\textbf{Experiment setup.}",
            r"\begin{itemize}",
            r"\setlength{\itemsep}{0.25em}",
            r"\setlength{\parsep}{0pt}",
            r"\setlength{\topsep}{0.25em}",
            r"\setlength{\partopsep}{0pt}",
            r"\item Phase08 run: \textbf{SmolVLA policy} rollouts.",
            r"\item Phase09 run: \textbf{oracle-action replay} baseline.",
            r"\item Metric: world-model latent distance $d\_goal\_l2\_wm\_int$.",
            r"\item Binning: env-action bins " + ", ".join(_expected_bin_labels(max_env_steps=int(payload["max_env_steps"]), stride=5)) + ".",
            r"\item For each task, rows are shown as \textbf{VLA} and \textbf{$\Delta_{\mathrm{oracle}}$} (oracle minus VLA).",
            r"\item Task names have the \texttt{-v3} suffix removed for readability.",
            r"\end{itemize}",
        ]
        if args.compare_format == "wide":
            tex = _emit_latex_compare_wide(
                bin_labels=bin_labels,
                compare_rows=payload["compare_csv_rows"],
                intro_lines=intro,
            )
        elif args.compare_format == "run-delta":
            tex = _emit_latex_compare_run_delta(
                bin_labels=bin_labels,
                compare_rows=payload["compare_csv_rows_run_delta"],
                intro_lines=intro,
                goal_frame_index=payload.get("goal_frame_index"),
            )
        elif args.compare_format == "summary":
            tex = _emit_latex_compare_summary(
                compare_rows=payload["compare_csv_rows_summary"],
                intro_lines=intro,
            )
        else:
            tex = _emit_latex_compare_long(
                compare_rows=payload["compare_csv_rows_long"],
                intro_lines=intro,
            )
        args.out_latex_compare.parent.mkdir(parents=True, exist_ok=True)
        args.out_latex_compare.write_text(tex, encoding="utf-8")

    print(
        f"[mt10_matrix] tasks={payload['n_tasks']} rows_csv={len(wide_rows)} "
        f"candidate_phase8={args.candidate_mode_phase8} candidate_phase9={args.candidate_mode_phase9}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
