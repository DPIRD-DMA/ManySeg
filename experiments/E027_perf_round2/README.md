# E027: performance round 2 (E026's leads)

Status: **done** (2026-10-22; RTX 4090, torch 2.14.1+cu130, code `618d147` + runtime patches).
Candidates from E026's lessons (Codex and Claude), measured as in E026: runtime patches
(`patches2.py`), current and patched runs alternating, equivalence checked first
(`check2.py`: float64 identical, bf16 error no larger than the current code's). Raw records in
`runs/`.

## Results

| candidate | measurement | result | applied |
|---|---|---|---|
| Compiled evaluation: own `torch.compile(model, dynamic=False)`, short last batch padded | compiled training steps (batch 16) then evaluation (batch 32, 500 images), 2nd epoch, 2 clean reps (`eval_compile_bench.py`) | PVTv2 U-Net 3584–3890 → **4617–4941 img/s (+22–27%)**, SegFormer 6318–6339 → **7094–7184 (+12–13%)**; first eval pass 3.9–5.8 → 3.1–3.8 s; padding alone (shared callable): no gain | yes (bench trainer) |
| Host arrays of device-resident datasets freed after the loaders are built | Sen1Floods11 train + val, process RSS (`ram_bench.py`) | 3.4 → 0.6 GB: **−2.8 GB host RAM for the whole run** | yes (bench runner) |
| U-Net `upsample="bilinear"` / `"aligned"`: output in the input dtype | train step, R34, 512², b16 | −3 to −4.5% time, −7% peak memory | yes |
| Late / same_extent intermediate concat fusion: resizes keep the dtype | train step, 4 + 2 bands (2nd at half resolution), b8 | −4% time; −4.5% memory (late), 0 (intermediate) | yes |
| Evaluator without host syncs (device sums, sync-free confusion matrix) | eager evaluation, 4 configs × 2 reps | 0 to −2% (evaluation is GPU-bound); metrics identical | no |
| ObjectMetrics: one device-to-host copy per batch | 8 × 256², 40 objects | 10.7–10.8 → 10.8–12.0 ms/batch | no |
| ViT neck: resize keeps the dtype | train step, ViT-S, 512² | 0 / 0 | no |

An eval-only benchmark first suggested a 5× gain from padding (`eval.jsonl`: compiled
evaluation with an unpadded last batch at 801–1248 img/s); in a training context the shared
callable is already near eager speed, and the gain is the +12–27% above.

The first pass of `eval_compile_bench` overlapped a CPU-heavy test run (`eval_compile.jsonl`);
the numbers above are the clean rerun (`eval_compile_clean.jsonl`).

## Files
`patches2.py`, `check2.py`, `eval_bench.py`, `eval_compile_bench.py`, `model_bench2.py` (E026's
`model_bench.py` with these patches), `objects_bench.py`, `ram_bench.py`, `run_e027.sh`,
`run_eval_compile*.sh`, `runs/*.jsonl`. Models come from `../E026_perf_audit/models.py`.
