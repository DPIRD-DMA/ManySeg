# E011c: Softplus density head on OAM-TCD (pre-registered)

Status: **complete** (registered 2026-10-29 at `663b51b` before any full run; 6 runs from a
frozen worktree at that commit on the shared local RTX 4090, no Pawsey units). Follow-up to
E011 (density arm) and the roadmap's Phase 5 row E011c. **Result: softplus passes the "no
worse" rule** (val +0.09 trees per tile, test +0.49, both well inside the margin). Count MAE
does not improve; the signed count bias moves closer to 0.

## Question
Should `manyseg.points.density_head()` default to `activation="softplus"`? On a downstream
drone dataset (few large plants per tile, one seed) softplus cut density count MAE from 10.1
to 2.9 against the raw head. OAM-TCD is the opposite regime: about 49 small trees per 1024²
tile, σ = 2 px. Softplus never reaches 0, so a background floor integrated over a 1024² tile
could cost counts here. This experiment measures whether it does.

## Protocol
Identical to E011's `density` arm except for the activation and the hardware:

* OAM-TCD CC BY subset at 1024² (20 cm); train = official folds 1–4, val = fold 0,
  test = official test. Canopy groups are ignore regions.
* ManySeg U-Net, resnet34 `tv_in1k`, recipe `fastai_like`, 30 epochs, batch 8, 512² random-
  resized crops (area 0.2–0.3) with flips; evaluation on full 1024² tiles. σ = 2 px, density
  target ×100 (`density_scale`), MSE loss weighted by the valid-pixel fraction.
* Arms (`--tag`), seeds 0, 1, 2 each, run interleaved by seed (plain s0, softplus s0, ...):

  | id | head | task kwargs |
  |---|---|---|
  | `density` | `density_head()`: raw output, bias 0 (current default) | `{"density": true}` |
  | `density_softplus` | `density_head(activation="softplus")`: bias −4 | `{"density": true, "density_activation": "softplus"}` |

* The plain arm is rerun locally rather than reusing E011's Setonix (MI250X) runs, so both arms
  share hardware and code. E011's numbers (val 20.02 ± 0.39, test 19.40 ± 0.37) are reported as
  a reproduction check only and play no part in the decision.
* Code: this branch adds `PointTask(density_activation=...)` (passed to `density_head`) and a
  `bias` field (mean signed count error) to count metrics. The plain arm's path is unchanged.
* Runs launch from a detached worktree at this commit, after a 1-epoch preflight of each arm
  through the same CLI, under a 12 GB systemd memory cap. `run.sh` / `launch.sh`.
* Budget: a 2-epoch timing run (softplus, seed 0, contended GPU) took 89 s for the frozen
  epoch and 74 s per unfrozen epoch, validation included; peak 4.0 GB VRAM. Estimate about
  39 min per run, about 4 h for 6 runs. The GPU is shared, so these are not timing claims.

## Metrics
* Primary: final-epoch **val count MAE** (trees per tile), as in E011's table.
* Secondary: val count RMSE and bias (mean predicted − true count), test MAE / RMSE / bias.
* Test is evaluated once at the end of each run (`--evaluate-test`). It does not choose
  anything; it is used only as the veto below.

## Decision rule
Let Δ = mean val MAE(`density_softplus`) − mean val MAE(`density`) over the 3 seeds, and
pooled SD = sqrt((SD_plain² + SD_softplus²) / 2) of the per-seed val MAEs (E011's definition).
The margin is m = max(2 × pooled SD, 0.025 × mean val MAE(`density`)); the second term (about
0.5 trees per tile) keeps the test meaningful if seed SDs come out very small.

* **Pass** ("no worse") if Δ ≤ m **and** test does not point the other way: test Δ ≤
  max(2 × pooled test SD, 0.025 × mean test MAE(`density`)).
* Pass: `density_head()` defaults to `activation="softplus"` (and the bench `PointTask`
  follows the library default). Changelog entry. Prepared on this branch, not merged.
* Fail: the default stays raw (`activation=None`); the `density_head` docstring records the
  OAM-TCD result next to the drone result.
* A softplus run that diverges or collapses counts with whatever final MAE it reaches; no
  reruns, no seed swaps.

## Preflight and visual check
One epoch of each arm (seed 0, models saved) ran through the same CLI from the frozen worktree
(`preflight/`). Rendered on val tiles 3, 100 and 400 and inspected before the full run:

* Target Gaussians sit on the labelled tree centroids, and each target sums to the tile's tree
  count (25, 32, 10).
* After the frozen first epoch the softplus head already peaks on trees, over a positive floor
  of about 7e-5 per pixel (in tree units). Integrated over a 1024² tile that floor is about 70
  trees, so the early softplus counts are far too high (count 69 on a 32-tree tile). The plain
  head is noise around 0 with negative values. Both behave as expected for one epoch. Whether
  the floor falls far enough by epoch 30 is what the full run measures.

## Results
Final epoch, seeds 0–2 (`uv run python experiments/E011c_softplus_density/analyse.py`).
Bias = mean (predicted − true) trees per tile; about 49 trees per val tile.

| arm | seed | val MAE | val RMSE | val bias | test MAE | test RMSE | test bias |
|---|---|---|---|---|---|---|---|
| density | 0 | 20.23 | 41.21 | −7.79 | 19.57 | 35.78 | −5.99 |
| density | 1 | 20.55 | 40.80 | +2.00 | 19.52 | 32.48 | +2.69 |
| density | 2 | 20.27 | 41.75 | −8.78 | 18.88 | 34.56 | −6.63 |
| density_softplus | 0 | 20.81 | 40.51 | −0.45 | 20.84 | 34.28 | +2.45 |
| density_softplus | 1 | 20.54 | 41.77 | +0.63 | 19.49 | 32.06 | +0.39 |
| density_softplus | 2 | 19.97 | 40.94 | −3.62 | 19.13 | 33.93 | −1.81 |

| arm | val MAE | test MAE | val bias | test bias |
|---|---|---|---|---|
| density | 20.35 ± 0.18 | **19.33 ± 0.38** | −4.86 | −3.31 |
| density_softplus | 20.44 ± 0.43 | 19.82 ± 0.90 | −1.15 | +0.34 |
| E011 density (Setonix MI250X) | 20.02 ± 0.39 | 19.40 ± 0.37 | | |

* Val: Δ = +0.09; pooled SD 0.33; margin max(0.66, 0.51) = 0.66. Within.
* Test (veto): Δ = +0.49; pooled SD 0.69; margin max(1.38, 0.48) = 1.38. Within.
* **Decision rule: pass.** Softplus is no worse on OAM-TCD.
* Reproduction: the local plain arm matches E011's Setonix runs (val 20.35 vs 20.02, test
  19.33 vs 19.40), within about one seed SD.
* No non-finite steps. All runs used 4.0 GB peak training memory. Runs took 23–35 min each
  and ran from 13:09 to 16:06. The GPU was shared throughout: another process held
  5.7–7.3 GB and every run's record says `contended`. These durations are not timing
  results.

## Interpretation
* Softplus neither helps nor hurts count MAE on OAM-TCD. The 3.5× gain on the drone data does
  not carry over. Both arms sit at about 20 trees per tile, with RMSE about twice MAE (a few
  tiles carry large errors). E011 put the large absolute error down to OAM-TCD's incomplete
  individual-tree labels; nothing here tests that.
* The background floor did not inflate counts after 30 epochs: softplus's mean signed error is
  closer to 0 than the plain head's (val −1.2 vs −4.9; test +0.3 vs −3.3). The plain head
  under-counts on 2 of 3 seeds, which fits negative outputs cancelling positive mass.
  Bias was a secondary metric with no decision attached, so this is descriptive.
* Softplus test MAE has a larger seed spread (0.90 vs 0.38), driven by seed 0 (20.84).
  With 3 seeds it is unclear whether that is a property of the arm.

## Decision
`density_head()` defaults to `activation="softplus"` (prepared on this branch, not merged).
It is no worse on OAM-TCD, its counts are positive by construction, and it was much better
on the drone data. The raw head stays available as `activation=None`.

## Deviations from the pre-registration
* None in protocol, arms, seeds or rule.
* After the runs, `run.sh`'s plain arm was made explicit (`"density_activation": null`), and
  E011's `run.sh` likewise, so both still reproduce the raw head under the new default. The
  runs above used the registered script, where the plain arm was the then-default raw head.
* The visual check could not be repeated on the final models: the full runs did not use
  `--save-model` (as in E011), so only the 1-epoch preflight models were rendered.
