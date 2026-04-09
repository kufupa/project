# Lessons from pi05-cube-FT audit

## What to keep
- Keep explicit oracle-export guardrails so failed rollouts are easy to detect without post-hoc labeling assumptions.
- Keep manifest validation checks when writing latent predictions, especially for tensor shape and key consistency.
- Keep explicit bridge bookkeeping for terminal status (`success`, `done`, and similar fields) so downstream evaluators can reason about episode termination.
- Keep deterministic PYTHONPATH ordering in launcher scripts and isolate compatibility overrides (`sitecustomize`, compatibility shims) behind explicit import order.
- Keep action-dimension sanity checks in dataset/eval adapters so dimension mismatches are surfaced before training starts.

## What not to keep
- Do not keep a *failed-only* oracle export policy as the default. It can hide success cases and makes failures hard to compare.
- Do not keep writing `latent_pred` manifests without validating that file entries match the expected data schema and export target.
- Do not keep bridge/export logic that drops `success`/`done` without an explicit replacement signal, since it can silently discard completion outcomes.
- Do not keep `PYTHONPATH` setup that appends compatibility paths after user/site paths by default; ordering should be explicit and controlled.
- Do not keep silent fallback from `action_dim=20` to `action_dim=4` without logging and hard-fail behavior.
- Do not keep hidden assumptions that fallback values are acceptable; require configuration and schema compatibility checks before execution.
