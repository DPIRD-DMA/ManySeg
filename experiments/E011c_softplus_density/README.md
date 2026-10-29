# E011c: Softplus density head on OAM-TCD (pre-registered)

Status: **registered** (2026-10-29, before any full run). Local RTX 4090 (shared), no Pawsey
units. Follow-up to E011 (density arm) and the roadmap's Phase 5 row E011c.

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
