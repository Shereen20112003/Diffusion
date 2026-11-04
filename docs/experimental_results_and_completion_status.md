# Experimental Results and Completion Status

Assessment date: 2026-11-01.

**2026-11-03 update:** The original-run assessment below is unchanged in scope.
A separate conditional metric-star implementation and smoke validation are
documented in the [conditional pilot status](conditional_metric_star_pilot_results.md).
Those smoke outputs do not complete the interrupted unconditional star study.

This is a read-only assessment of `experiments/results/remaining_full_20261101`,
not a claim that the full study completed. No training was restarted.
The source artifacts and generated samples were left unchanged.

## What completed

| Dataset | Completed independent training seeds | Status |
|---|---|---|
| Two moons | 100, 101, 102, 103, 104 | All five completed, including three landmark rules, four MLP losses, and kernel/learned Frechet decoders. |
| Eight Gaussians | 100, 101, 102, 103, 104 | All five completed with the same comparisons. |
| Sphere | 100 | One of five completed; seed 101 has only a partial attempt. |
| Fixed metric star | None | No full-study result; earlier smoke tests are not scientific replications. |
| Distribution-valued objects | None | No full-study result; earlier smoke tests are not scientific replications. |

There are **11 of 25 completed dataset/seed units**, **645 of 855 planned
metric rows**, and 47 aggregate groups. The two planar datasets account for
630 rows, and sphere seed 100 for 15. Row counts do not measure remaining
runtime: the single completed sphere unit took about 3.67 hours.

The root `completion.json` is absent. The recorded failure is
`OSError: [Errno 22] Invalid argument` in the progress-print function while
announcing the sphere seed 101 direct baseline. This is a console/output
failure, not evidence of failure in the spherical solver. Its timing is near
the stop request, but the log does not establish causality. Completed outputs
are preserved, and the partial seed is excluded from the results below.

## Five-seed planar results

The [complete seed-level results](two_moons_and_eight_gaussians_experimental_results.csv)
include seeds 100-104 separately for all 21 method/landmark configurations
per dataset, together with full-precision means and between-seed SDs.

Shareable PNG tables are available for the
[combined mean and SD results](figures/two_moons_and_eight_gaussians_results.png),
[two-moons seed-level results](figures/two_moons_independent_run_results.png),
and [eight-Gaussian seed-level results](figures/eight_gaussians_independent_run_results.png).

Graphical comparisons are available as a
[complete mean-and-SD plot](figures/two_moons_and_eight_gaussians_method_comparison.png),
[two-moons independent-run plot](figures/two_moons_independent_run_comparison.png),
and [eight-Gaussian independent-run plot](figures/eight_gaussians_independent_run_comparison.png).
The independent-run plots use the primary embedded-MSE MLP and both
Frechet decoders for readability; the complete plot includes all MLP losses.
Error bars denote one between-run SD, not confidence intervals.

Entries are **mean W2 +/- between-training-seed standard deviation**, not
standard errors or confidence intervals. For each model, three fresh
512-object evaluation draws are first averaged within each training seed.
Means and sample SDs are then computed across the five seeds. Thus 15
evaluation rows do not represent 15 independent trained models.

| Method | Landmarks | Two moons | Eight Gaussians |
|---|---|---:|---:|
| Direct DDPM | Not applicable | 0.1788 +/- 0.0250 | 0.4440 +/- 0.1010 |
| Empirical training-object resampling | Not applicable | 0.1493 +/- 0.0086 | 0.3838 +/- 0.0217 |
| Real-real comparison | Not applicable | 0.1454 +/- 0.0318 | 0.3567 +/- 0.0498 |
| Coordinate DDPM + embedded-MSE MLP | Manual | 0.1815 +/- 0.0195 | 0.3759 +/- 0.0203 |
| Coordinate DDPM + embedded-MSE MLP | Random | 0.1593 +/- 0.0226 | 0.4071 +/- 0.0416 |
| Coordinate DDPM + embedded-MSE MLP | FPS | 0.1823 +/- 0.0403 | 0.4248 +/- 0.0397 |
| Coordinate DDPM + kernel Frechet | Manual | 0.1821 +/- 0.0202 | 0.3874 +/- 0.0191 |
| Coordinate DDPM + kernel Frechet | Random | 0.1572 +/- 0.0234 | 0.4209 +/- 0.0415 |
| Coordinate DDPM + kernel Frechet | FPS | 0.1805 +/- 0.0406 | 0.4327 +/- 0.0379 |
| Coordinate DDPM + learned Frechet | Manual | 0.1794 +/- 0.0214 | 0.3756 +/- 0.0203 |
| Coordinate DDPM + learned Frechet | Random | 0.1552 +/- 0.0235 | 0.4075 +/- 0.0411 |
| Coordinate DDPM + learned Frechet | FPS | 0.1785 +/- 0.0400 | 0.4249 +/- 0.0406 |

All landmark rules are shown, rather than reporting only the best observed
test result. Random landmarks have the lowest observed coordinate-method
means for moons; manual landmarks do for eight Gaussians. This is not a
test-selected deployment rule or evidence of uniform dominance.

The four MLP losses have mean W2 differences below 0.0003 within each
dataset/landmark configuration. These runs do not support a material W2
advantage for Huber or standardized losses. The complete loss-specific
results remain in the source `summary.csv` and `summary.json`.

Empirical resampling is competitive and has a lower observed moons W2 than
all fitted generators. It must remain visible in the comparison.
The real-real number is finite-sample context, not a strict lower bound.
The displayed SDs alone do not establish a statistically significant
improvement; paired seed-level comparisons and broader replication are
needed before making superiority claims.

## The imperfect two-moons shapes are a real result

The stored plots retain between-arc points and off-arc samples.
For the primary MLP decoder, the five-seed average fraction farther than
0.15 from either clean arc is:

| Method | Far-from-arc fraction | Mean distance to clean arcs |
|---|---:|---:|
| Direct DDPM | 5.48% | 0.0538 |
| MLP, manual landmarks | 9.99% | 0.0663 |
| MLP, random landmarks | 9.51% | 0.0634 |
| MLP, FPS landmarks | 10.21% | 0.0663 |
| Kernel Frechet, manual / random / FPS | 3.20% / 2.75% / 3.24% | 0.0522 / 0.0397 / 0.0533 |
| Learned Frechet, manual / random / FPS | 7.02% / 6.02% / 6.84% | 0.0534 / 0.0531 / 0.0533 |

These thresholds are geometry diagnostics, not invalid-object criteria:
the true data include Gaussian noise. Kernel reconstruction can reduce
off-arc distances while producing discrete-looking clumps around reference
objects. Neither closeness to the noiseless arcs nor a lower W2 alone proves
faithful recovery of the noisy target.

For example, the stored plots at
`two_moons/seed_100/attempt_1/manual/comparison.png` and
`two_moons/seed_104/attempt_1/fps/comparison.png` show those tradeoffs.
They are illustrations, not substitutes for the five-seed aggregates.
There was no sample filtering, plot retouching, or retraining to improve them.

For eight Gaussians, the inspected seed-100/manual comparison shows all eight
clusters in the direct and coordinate generators, with residual outliers.
Across seeds, the direct baseline's mean distance to its nearest true center
is 0.2122; the primary MLP ranges from 0.2110 to 0.2181 across landmark rules.
The fraction beyond 0.45 from a center is 6.63% for direct DDPM and
6.93%-7.73% for the MLP. Thus its lower W2 does not imply fewer off-cluster
samples. The original blurred-ring concern improved, but perfect geometric
and mixture-probability recovery is not established by the inspected plot.

## Sphere: one preliminary seed only

| Method | W2, average of three evaluation draws |
|---|---:|
| Ambient DDPM + radial normalization | 0.1129 |
| Coordinate DDPM + kernel Frechet | 0.1381 |
| Coordinate DDPM + learned Frechet | 0.1380 |
| Empirical resampling | 0.1263 |
| Real-real comparison | 0.1033 |

The direct baseline has lower observed W2 in this seed. There is no
between-training-seed SD available yet. Both Frechet methods have zero
observed fraction outside the enclosing data cap; the normalized direct
baseline has 0.78125% outside that cap. Points outside the cap can still be
valid sphere-valued objects.

For the two Frechet decoders, the across-draw average of the maximum
weighted-log residual is approximately 2.64e-7 and 2.80e-7, respectively.
These are stationarity diagnostics, not certified global objective gaps.
The current direct baseline is not an intrinsic Riemannian diffusion
reproduction.

## Provenance and remaining scope

The counts above use actual dataset/seed `completion.json` markers.
All 47 stored W2 mean/SD summaries were independently reconstructed from
the 645 raw rows; their maximum absolute discrepancy was about 1.11e-16.
Partial attempts and smoke-test outputs were not included.

Remaining execution includes four sphere seeds and all five seeds each
for the fixed-star and distribution-valued studies. The literature review
also identifies missing specialized external baselines; these are separate
from finishing the original runner. Conditional generation remains future
work. None of these stopped jobs was resumed.

These new full-study artifacts and this report are local; they have not
been pushed to GitHub. See the
[literature assessment](conditional_generation_of_non_euclidean_data_literature_review.md)
for prioritized, geometry-matched baselines.
