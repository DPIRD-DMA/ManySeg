# Experiments

The evidence behind ManySeg's defaults, tips and the README's evidence table, negative results
included. Each `E*/` folder is one experiment: its `README.md` states the question, the plan and
decision rule (most were registered before the runs), the results and the conclusion, next to
the scripts that ran it and, in `runs/`, the JSON run records.

The write-ups are also on the [documentation site](https://manyseg.readthedocs.io), under
Experiments.

* Finished experiments are records: corrections go in an amendment, not over the results.
* Most runs use the `manyseg-bench` package (`packages/manyseg-bench`) for data, training and
  evaluation. Scripts read data from `MANYSEG_BENCH_DATA`; some also need `MANYSEG_RUNS`
  (frozen worktrees) or `MANYSEG_REPO`. Recorded paths use the same names.
* Some experiments ran on older PyTorch versions or other hardware; each write-up says which.
  Compare numbers within an experiment.
* `analyse_runs.py` summarises run records; `queues/` holds the shell queues that ran several
  experiments back to back.
