# Tripod Decoder Diagnostic: Results and Meeting Notes

Date: 2026-11-04. Experiment: `conditional_tripod_decoder_only_debug`.

**Bottom line:** We reproduced the original discrepancy and identified its
mechanism in the frozen seed-100 example. At the original bandwidth, **none of
the 13 true centers survived**: all moved a short distance onto arm 1. The two
decoded centers were **false positives from noncenter inputs**, not retained
true centers. This is a reconstruction effect with true coordinates, not a
DDPM error.

## 1. A one-minute update for Changjun

> I ran the decoder alone with true coordinates, keeping the previous encoder
> and reference sample fixed. The original 13-center-to-2-center observation
> actually means zero true centers were retained and two noncenters became
> centers. Every true center maps to the same point on arm 1, at radius
> 0.00366235.
>
> The reason is local reference imbalance: at the encoded center, arm 1 has
> more Gaussian-weighted radial moment than the other two arms combined. The
> exact Fréchet rule therefore moves the output off the center. This happens
> even with the controlled injective encoder.
>
> A balanced reconstruction dictionary restores true-center retention in this
> example, without changing the encoder or absolute bandwidth. But it also
> creates false centers near the vertex in the no-atom scenario. Thresholding
> has the same tradeoff and cannot remove false centers already present.
>
> This supports separating identification by the encoder from accuracy of the
> chosen decoder. I suggest resolving that decoder tradeoff before increasing
> diffusion training.

These are **one-reference-seed debugging results**, not a benchmark, a general
decoder fix, or evidence of method superiority.

## 2. What was run

Following the supplied
[decoder debugging manual](../Shereen_Tripod_Decoder_Debug_Manual_20261104.md):

- No DDPM training or sampling.
- Scenarios: `S0_no_atom`, `S1_varying_atom`,
  `S2_near_vertex_no_atom`.
- Independent held-out true samples at **X=0.1, 0.5, 0.9**.
- **2048 responses per X per scenario**, totaling **18432** new observations.
- Original **48 reference responses** from seed 100, including repeated
  centers, reused unchanged.
- The original sample-medoid base, three FPS landmarks, coordinate centering
  and scales reconstructed and checked against the saved configuration.
- Gaussian ratios **h/h_scale = 0.05, 0.10, 0.20, 0.40**.
- Raw outputs and full-vertex thresholds
  **a = 0, 0.0025, 0.005, 0.01, 0.02, 0.05**.
- Frozen primary encoder and a separately labeled controlled
  center/endpoints encoder.
- Original B versus an explicitly oracle, symmetrized B diagnostic.

All bandwidth and threshold comparisons reuse the same observations. Changing
B does not refit the encoder. Its bandwidth scale remains the one from
**original B within that encoder**, so changing B does not silently change h.

The run completed in **9.57 seconds**, produced **864 per-X metric rows**, and
passed **48 identical-center audits**. All **49 repository tests** passed.

## 3. Correct interpretation of the original 27.08% to 4.17% result

The original 48 held-out observations were audited again with the original
primary encoder and original B.

| h/h_scale | True centers retained (TP) | True centers lost (FN) | Noncenters mapped to center (FP) | Noncenters retained off-center (TN) | Where an exact center goes |
|---|---:|---:|---:|---:|---|
| 0.05 | 0 | 13 | 0 | 35 | Arm 2, radius 0.00115896 |
| **0.10** | **0** | **13** | **2** | **33** | **Arm 1, radius 0.00366235** |
| 0.20 | 13 | 0 | 0 | 35 | Center |
| 0.40 | 13 | 0 | 2 | 33 | Center |

At ratio 0.10, MSE exactly reproduces the earlier value:
**0.0009427665835979077**.

Thus, the earlier aggregate percentages did not measure atom retention.
At ratio 0.10 the actual retention was **0/13**, not 2/13.

## 4. Why does the center move?

The reference branch counts are:

| Scenario | Center copies | Arm 1 | Arm 2 | Arm 3 |
|---|---:|---:|---:|---:|
| S0 | 0 | 14 | 20 | 14 |
| S1 | 7 | 13 | 15 | 13 |
| S2 | 0 | 14 | 19 | 15 |

Counts alone are not the determining quantity. The exact decoder depends on
**Gaussian-weighted radial moments evaluated at E(center)**, which also depend
on the radii, coordinate geometry, scaling, and bandwidth.

For S1, the minimum positive reference radii on the three arms are
**0.07871, 0.05577, 0.22704**. Arm 3 has no references comparably close to the
center. Even though its total arm count matches arm 1, its local contribution
can be much smaller.

At the original ratio **0.10**, `h_scale=2.24933673` and `h=0.22493367`:

| Quantity at E(center) | Original B | Balanced oracle B |
|---|---:|---:|
| T1 | 0.00873923 | 0.00460537 |
| T2 | 0.00490293 | 0.00460537 |
| T3 | 0.00017395 | 0.00460537 |
| g1 = T1 - T2 - T3 | **0.00366235** | -0.00460537 |
| g2 | -0.00401025 | -0.00460537 |
| g3 | -0.01346820 | -0.00460537 |
| Decoded center input | Arm 1 | Center |

The exact rule is:

- If all g values are nonpositive, return the center.
- If one is positive, return that arm with radius equal to its positive g.

Here g1 is positive by approximately 0.00366, **far above the 1e-12
numerical-audit tolerance**. This is not a tiny-sign solver artifact.
The decoder is analytic; no iterative solver is involved.

### Balanced dictionary construction and its limitations

For each positive radius in original B, place a copy at that radius on all
three arms, with prior weight **1/3 per copy**. Keep each original center copy
with prior weight 1.

This preserves the total reference prior mass, center prior mass, and pooled
radial distribution, while making branchwise radial coverage identical. It
changes B only; A, the base, preprocessing, data and absolute h remain fixed.

The raw dictionary size increases: S1 has **130 entries**, not 48, but total
prior mass remains 48 and center prior mass remains 7. S0/S2 have 144 entries
with total prior mass 48. This is an **oracle intervention to diagnose
imbalance**, not a new default decoder, an equal-compute learned baseline, or
a claim that balancing solves every reconstruction problem.

## 5. New held-out conditional results

### S1 at X=0.5: bandwidth alone

Same 2048 true observations, primary encoder, original B, raw decoder.
There were **704 true centers** in this realization. The analytic center
probability is **0.35**; the sample center fraction is **704/2048 = 0.34375**.

| h/h_scale | TP | FN | FP | TN | Reconstruction MSE |
|---|---:|---:|---:|---:|---:|
| 0.05 | 0 | 704 | 2 | 1342 | 0.001068 |
| 0.10 | 0 | 704 | 6 | 1338 | **0.000914** |
| 0.20 | 704 | 0 | 6 | 1338 | 0.001520 |
| 0.40 | 704 | 0 | 24 | 1320 | 0.007462 |

The smallest MSE in this table does **not** preserve a single true center.
Increasing bandwidth can restore the atom but also increase smoothing and
false positives. No bandwidth is selected on these test results.

### S2 at X=0.5: false centers and thresholding

S2 has **no true center atom**, and none of these 2048 inputs is exactly central.
At ratio **0.10**:

| Reconstruction / postprocessing | False centers | False center mass |
|---|---:|---:|
| Original B, raw | 117 | 5.71% |
| Original B, a=0.005 | 355 | 17.33% |
| Original B, a=0.01 | 449 | 21.92% |
| Balanced oracle B, raw | 353 | 17.24% |
| Balanced oracle B, a=0.01 | 485 | 23.68% |

Balancing restores true-center retention in S1 but can **worsen false-center
mass in S2** by creating a region of noncenter inputs that reconstruct to the
center. It is not a universal repair.

For perfect, undecoded S2 samples, the population mass snapped to the center
at a=0.01 is **11.4625%**. That is a separate baseline, not a number to subtract
from decoder error.

Thresholding can only preserve or increase an existing center mass; it cannot
remove the 117 false centers already present in the raw S2 output.

### S1 threshold benefit, with its cost

At X=0.5 and ratio 0.10, threshold a=0.005 recovers all 704 displaced
true centers because their common radius is 0.00366235. It also increases
false positives from **6 to 18**. At a=0.01, false positives rise to 25.
This illustrates why raw retention, false positives, and the threshold
sensitivity must be reported separately.

S0 likewise produces false raw centers: at X=0.5 and ratio 0.10, **5/2048**
noncentral responses reconstruct to the center.

## 6. The identical-center implementation check

For fixed E and D, every center input has the same encoded coordinate.
The center inputs therefore have one common decoded object, independent of X.

All **48 configuration audits passed**:

- One unique encoded coordinate for repeated center probes.
- One unique encoded coordinate among actual true-center inputs where present.
- Consistent raw center/noncenter labels for all true centers.
- Maximum cross-batch floating-point output discrepancy:
  **9.71e-17**, below the explicit 1e-12 audit tolerance.

The tolerance is used to audit numerical agreement only. **Raw center labels
still require radius exactly zero**. No positive radius is silently snapped.

## 7. Identification versus reconstruction: what to explain

### Identification asks whether the information exists

If E is injective on the relevant support and has the usual measurable-inverse
conditions, knowing the exact encoded law identifies the original law.
This is a property of the representation, not a guarantee about a particular
practical decoding algorithm.

For the controlled tripod encoder, the representation is explicitly
invertible: the norm of the normalized coordinate gives the radius, and its
largest component identifies the arm for a noncentral point.

As an independent sanity control, this analytic inverse recovered all **18432**
new observations to maximum coordinate error **3.33e-16** and preserved center
labels exactly. The Gaussian Fréchet decoder can nevertheless distort those
same true coordinates. This isolates an issue with **that reconstruction
choice**, not a demonstrated lack of information in the controlled encoding.

### Reconstruction asks what the chosen algorithm does with the information

The Gaussian Fréchet decoder is a locally weighted average over B, not the
analytic inverse of E. Its output can shift a center toward one arm or collapse
near-center observations to the center. Injectivity does not force
`D(E(y))=y`.

This also explains why MSE can be small while atom probabilities are wrong:
moving an entire center atom only 0.00366 away has little distance cost, but
its probability of being **exactly** at the center becomes zero.

### A useful wording clarification about "isometric"

The existing controlled code uses
`E(j,r)=r(2e_j-1)/sqrt(3)` with a **Euclidean Gaussian kernel**. This is
injective and bi-Lipschitz, but **not exactly Euclidean-isometric**:
two radius-0.4 points on different arms have intrinsic distance 0.8, while their
encoded Euclidean distance is `0.8*sqrt(2/3)`.

The unnormalized contrast vector is isometric with the **supremum norm** on
these tripod coordinates. Thus an isometry claim should specify the norm and
normalization. This clarification does not weaken the injectivity argument.

### Relation to the new corollary

Changjun's stated coverage result provides approximate identification under its
landmark-net assumptions; injectivity is not the only route to identification.
It is a separate statement from accuracy of Gaussian Fréchet reconstruction.
The newly mentioned rate corollary has not been supplied or independently
verified here. This diagnostic neither proves nor refutes it.

No manuscript proof has been changed. Base-point sensitivity and larger
DDPM/label-aware-baseline experiments remain deferred.

## 8. Pictures to show, in order

1. [Exact center displacement](../experiments/results/decoder_debug_20261104/exact_center_displacement.png):
   directly shows where the center goes and what balancing changes.
2. [S1 center mass versus bandwidth](../experiments/results/decoder_debug_20261104/S1_center_mass_vs_bandwidth.png):
   top row original B; bottom row balanced B; raw and thresholded curves.
3. [S2 false center mass versus bandwidth](../experiments/results/decoder_debug_20261104/S2_false_center_mass_vs_bandwidth.png):
   shows why neither balancing nor snapping is a universal fix.
4. [Reference counts and radial coverage](../experiments/results/decoder_debug_20261104/reference_balance.png):
   use to explain why branch counts alone are insufficient.
5. [Raw false-positive and false-negative mass](../experiments/results/decoder_debug_20261104/raw_center_confusion.png):
   separates genuine atom loss from spurious center creation.

The curves have one reference seed and no between-seed uncertainty bands.
Different X values share a fixed decoder but have different input laws.

## 9. Reproducibility package

- [Full output directory](../experiments/results/decoder_debug_20261104/)
- [Run configuration, Git HEAD/dirty state, and code hashes](../experiments/results/decoder_debug_20261104/config.json)
- [Completion marker](../experiments/results/decoder_debug_20261104/completion.json)
- [Independent post-run verification and analytic inverse check](../experiments/results/decoder_debug_20261104/verification.json)
- [49-test regression log](../experiments/results/decoder_debug_20261104/test_log.txt)
- [Per-X confusion matrices, MSE, arm errors, bandwidths and thresholds](../experiments/results/decoder_debug_20261104/per_x_decoder_confusion.csv)
- [Exact center branch moments and sign margins](../experiments/results/decoder_debug_20261104/center_branch_balance.csv)
- [Original 48-observation audit](../experiments/results/decoder_debug_20261104/original_48_observation_audit.csv)
- [Reference counts, prior masses, and radial summaries](../experiments/results/decoder_debug_20261104/reference_balance.csv)
- [Identical-center audits](../experiments/results/decoder_debug_20261104/identical_center_audits.csv)
- [Perfect-sample threshold-bias control](../experiments/results/decoder_debug_20261104/perfect_sample_threshold_bias.csv)
- [Diagnostic runner](../experiments/diagnose_star_decoder.py)
- [New focused tests](../experiments/test_star_decoder_debug.py)

The implementation is uncommitted. The run records the existing Git HEAD plus
dirty-worktree state and exact source SHA-256 hashes; it does not pretend the
new code is already represented by a committed revision.

To reproduce with a new empty output directory:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s .\experiments -p "test_*.py"
& .\.venv\Scripts\python.exe .\experiments\diagnose_star_decoder.py `
  --samples-per-x 2048 --output .\experiments\results\decoder_debug_repeat
```

## 10. Decision for the meeting

The next question is not simply "train longer." It is:

> How should reconstruction retain a genuine vertex atom without collapsing
> a continuous near-vertex population into a false atom?

The controlled inverse is a useful diagnostic on this known tripod, not a
generic metric-space reconstruction proposal. Discuss decoder-specific
assumptions or modifications, then validate them on both S1 and the S2 negative
control before returning to larger conditional DDPM experiments.
