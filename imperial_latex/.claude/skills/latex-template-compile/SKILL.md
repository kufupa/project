---
name: latex-template-compile
description: Use when compiling Imperial College LaTeX template, fixing main.tex build failures, or chasing latexmk, pdflatex, bibtex, main.pdf, EPS, or Ghostscript errors.
allowed-tools: Read Write Edit Bash
---

# LaTeX Template Compile

## Regenerate figures (CPU only, before PDF)

```bash
cd Imperial_College_Individual_Project_Template
module load tools/prod matplotlib/3.10.5-gfbf-2025b
python3 scripts/generate_thesis_figures.py
```

Writes `figures/*.pdf` from on-disk `eval_sweep_summary.json` under `project/artifacts/`.

## RDS / HPC (preferred when login compile is slow)

```bash
cd Imperial_College_Individual_Project_Template
mkdir -p logs/pbs/thesis
qsub scripts/compile_main.pbs
qstat -u "$USER"
```

- PDF: `build/main.pdf`
- Log: `logs/pbs/thesis/compile_main.out`
- Compiles in template root on a CPU node (reuses `main.aux`), then `cp main.pdf build/main.pdf`.

## Quick build (login node)

Fine for small incremental edits if it finishes in ~1–2 minutes:

```bash
cd Imperial_College_Individual_Project_Template
module load texlive/20230313-GCC-13.2.0
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
cp -f main.pdf build/main.pdf
```

Do **not** use `-output-directory=build` — cold-starts aux and is very slow on NFS.

## If build fails

- Read `main.log` or `logs/pbs/thesis/compile_main.out`.
- `No file main.bbl` on first run is normal; `latexmk` runs BibTeX.
- Missing `gs` for `title/logo.eps`: use `title/logo.pdf` or the text fallback in `title/title.tex`.

## Files to check

- `main.tex`, `title/title.tex`
- `main.log`, `build/main.pdf`
- `logs/pbs/thesis/compile_main.out`
