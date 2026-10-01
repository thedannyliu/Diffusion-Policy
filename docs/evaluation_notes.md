# Evaluation definitions and evidence status

## Metrics

- `test_mean_score`: mean across episodes of maximum reward. PushT clips coverage divided by its success threshold to [0, 1].
- `test_target_area_coverage`: mean of each episode's maximum recorded target coverage, not final coverage.
- `test_success_rate`: future runs aggregate the environment's explicit `success` history. PushT succeeds when coverage exceeds 0.95. Wrapper termination also includes the time limit and is not task success.
- `inference_latency_p50_ms`: median recorded action-sampler latency, excluding observation encoding and action postprocessing. It is not a real-robot closed-loop period; the saved JSON omits complete hardware and batch/timing provenance.
- `test_smoothness`: the runner's trajectory smoothness statistic. It should not be treated as a physical jerk measurement without checking units and differencing conventions.

## Historical success-rate correction

The old runner used the maximum wrapper `done` flag per episode. `MultiStepWrapper` sets this flag when the step limit is reached, so unsuccessful timeouts also counted as successes. This explains why even the poor FM Transformer run reports 100% in the committed summaries. Those values are invalid as task-success rates.

The runner now uses the `success` field emitted by `PushTEnv` and retained by `MultiStepWrapper.get_infos()`. Missing or empty histories raise an error. This change does not rewrite historical JSON or invent corrected rates. Re-evaluation from the original checkpoints is required for fresh task-success rates.

## Scope of numerical comparisons

The README table is traceable to `results/*/eval_results.json`. `scripts/summarize_results.py` reads those files directly and excludes historical success rates. The lower-learning-rate FM run has 97.8% score retention and approximately 27.45x p50 speedup relative to the DDPM run.

Training-time records include validation rollouts and different inference budgets; they do not isolate the cost of the training objective. Most configurations use one training seed. Report uncertainty only after matched multi-seed runs.

## Real-world evidence to reconcile

The previous root README reports:

| Setup | Cube | Sphere |
| --- | --- | --- |
| Two-camera, interior | 32/32 | 32/32 |
| One-camera, interior | 30/32 | 16/32 |
| Two-camera, boundary | 16/19 | 11/19 |

Its boundary header says 18 positions, and the described collection grid has 32 interior plus 18 boundary positions. Trial-level records are absent from this checkout. Keep the discrepancy explicit until the original experiment records establish whether there were additional/repeated trials. Do not infer an overall rate or silently change a denominator.

The real-robot runs use DDPM. The flow-matching results use PushT simulation. Project-level implementation and experimental work should be distinguished from each team member's individual contribution when presenting this project.

## Missing source restoration

The checkout imported `PushTImageEnv` but did not contain its source file. The root ignore pattern `env/` also matched the vendored source directory. It is now restricted to `/env/`, and `pusht_image_env.py` is restored unchanged from the upstream Diffusion Policy source, under the existing vendored MIT license. The lightweight metric tests do not validate the full legacy simulation environment.
