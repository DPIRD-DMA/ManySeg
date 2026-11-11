# Contributing

Bug reports and pull requests are welcome. For a new feature, open an issue first: new options
need a test or an experiment behind them (see [`docs/roadmap.md`](docs/roadmap.md)).

## Setup

ManySeg uses [uv](https://docs.astral.sh/uv/). The repository holds two packages: `manyseg`
(the library) and `manyseg-bench` (the benchmark). The library must not import the benchmark.

```bash
uv sync --all-packages
```

## Before opening a pull request

```bash
uv run pytest packages/manyseg -n 4            # fast tests
uv run pytest packages/manyseg-bench -n 4
uv run pytest packages/manyseg -m slow -n 4    # compile, export, ONNX: run if you touched these
uvx ruff check packages examples notebooks scripts experiments/analyse_runs.py
python scripts/sync_readme.py                  # if you edited README.md
```

Add a regression test with every bug fix, and list user-visible changes in
[`CHANGELOG.md`](CHANGELOG.md). [`docs/testing_strategy.md`](docs/testing_strategy.md) explains
the test tiers.

## Experiments

Each `experiments/E*/` folder records one experiment: its plan, decision rule, code and results.
Scripts read data from `MANYSEG_BENCH_DATA`; some also need `MANYSEG_RUNS` (frozen worktrees) or
`MANYSEG_REPO`. Recorded paths in `runs/` use the same names, plus `$WORKSPACE` for the parent
directory of the checkouts. Treat finished experiments as records: add an amendment rather than
changing their results. Commit hashes in older write-ups and run records predate a history
rewrite; [`experiments/COMMIT_MAP.md`](experiments/COMMIT_MAP.md) maps them to this repository's.
