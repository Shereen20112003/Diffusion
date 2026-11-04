# Conditional tripod pilot — decoder debugging handoff

**Purpose:** Find out *why the Gaussian Fréchet reconstruction changes the probability of the center vertex*, without confusing that problem with DDPM training. This is a development diagnostic, not a benchmark or proof of model superiority.

**What we know from the reported smoke run (not independently verified from code):** one training seed, 128 training pairs, 40 optimizer updates, 12 sampling steps, three test covariate values. In the S1 held-out *true-coordinate* check, 13/48 original responses (27.08%) were at the center but only 2/48 decoded outputs (4.17%) were at the center; intrinsic reconstruction MSE was 0.000943. The two percentages are sample fractions, **not** estimates precise enough to establish population performance. In S2, snapping even perfect true samples at distance 0.01 creates 11.4625% false center mass by design.

## Priority 1 — Isolate the decoder; no additional DDPM training yet

1. Freeze **one exact set of references** (base `o`, encoding landmarks `A`, reconstruction objects `B`), fitted coordinate centering/scales and the Gaussian bandwidth `h`. Save those settings and the random seed.
2. Draw independent true responses `Y` at a few fixed covariate values, such as `x = 0.1, 0.5, 0.9`, under S0, S1 and S2. Use at least 1,000 true responses per `x` for this diagnostic if computationally feasible (not 48 pooled responses). These samples do **not** train a model.
3. For each true response, calculate the **true** encoded coordinate `Z = E(Y)`. Decode with the *existing* Gaussian Fréchet decoder to obtain `Y_raw = D_h(Z)`; do not run the DDPM.
4. Record the intrinsic graph-distance squared error, the true center count, the *decoded* center count, and the four center-confusion counts. Report results **separately for each x**, not just one pooled percentage.

| True label | Decoded center | Decoded off center |
|---|---|---|
| Center | TP: correctly preserved atoms | FN: true atoms lost |
| Not center | FP: spurious center atoms | TN: correctly not at center |

**Important implementation check.** With a **shared, fixed encoder and deterministic exact decoder**, every true center observation `Y=v` has the same encoded coordinate `E(v)` and must decode to the **same** point. Consequently, all true centers in that frozen model are either preserved or displaced. The reported 2/48 decoded centers could be *false positives* from noncenter points; the two aggregate percentages cannot tell. If true center inputs have inconsistent outputs under identical references, check whether references, scales, weights, solver tolerances, or model states accidentally change across batches or covariates.

## Priority 2 — Print the exact branch-balance diagnostic at the center

On the tripod, each reconstruction reference `b_l` has an arm label `j_l` (or center label `0`) and its distance `r_l` from the center. Compute **Gaussian weights evaluated at the true center coordinate `E(v)`**. For each arm `j=1,2,3`:

```
T_j = sum_l [weight_l(E(v)) * r_l * 1{j_l = j}]
T   = T_1 + T_2 + T_3
g_j = 2*T_j - T

if max(g_1,g_2,g_3) <= 0:
    decode_to_center()
else:
    j_star = argmax_j g_j
    decode_to_arm(j_star, radius=g_j_star)
```

- Print `T_1,T_2,T_3`, `g_1,g_2,g_3`, `max(g)`, the decoded arm/radius, `h`, the number of references on each arm, the number at the center, and minimum positive reference radius. Center references have radius zero.
- **The exact decoder returns the center if and only if `max(g) <= 0`.** A positive `max(g)` identifies which arm receives a displaced true-center point. Very small values near zero warrant a numerical-precision check.
- Use the true intrinsic tripod distances for all computations and object-space W2. A planar drawing is only for visualization and for the separately labeled ambient baseline.
- Report numerical tolerance explicitly. Do not silently convert a positive output radius into an exact center atom.

## Priority 3 — One-variable-at-a-time sensitivity study

**First vary bandwidth only, holding `o,A,B`, data and seeds fixed.** Record the current bandwidth and test a small relative grid, e.g. `h/h_scale = {0.05, 0.10, 0.20, 0.40}`, where `h_scale` is the median *positive* pairwise distance among encoded reference objects. If the current implementation uses another bandwidth scale, retain and report it rather than replacing it silently.

For each `h`, report at each `x`:

- Intrinsic reconstruction MSE; center TP/FN/FP/TN; raw vertex-mass error; and three arm-probability errors.
- `g_1(E(v)), g_2(E(v)), g_3(E(v))`; reference counts and radial weighted masses by arm.
- Separate S0 (no center atom), S1 (true center atom), and S2 (near-center but no atom).

**Next compare reference configurations at the same bandwidth scale:** (a) the existing reference selection and (b) a **controlled balanced diagnostic dictionary** with comparable radial coverage on all three arms. The balanced dictionary is an *oracle diagnostic* to locate the issue, not a new default method or a fair learned baseline. Keep other factors fixed. Check the total reference counts and possible duplicated center objects.

**Only afterward** consider base-point sensitivity: squared-distance sample medoid (default) versus empirical Fréchet mean, when it exists and satisfies the encoder's admissibility conditions. If changing `o` forces FPS landmarks to change, record that fact: it is no longer a base-only ablation.

## Priority 4 — Thresholding: report both benefits and false atoms

For the **same decoded samples**, compare the raw output (`a=0`) and prespecified snapping radii `a = 0.0025, 0.005, 0.01, 0.02, 0.05`. Calculate the true population center fraction separately from its finite evaluation-sample realization.

- **S1:** Does snapping recover true center probability without excessive false positives?
- **S0 / S2:** Does it create center atoms where the true law has none?
- Also apply exactly the same snapping rules to **perfect true samples** as an unavoidable-bias diagnostic; S2 already has 11.4625% near-center mass at `a=0.01`.
- Snapping **cannot reduce** the center atom already present in raw outputs; it can only preserve or increase it. Thus it cannot fix spurious raw center mass in S2.
- Never choose the "best" threshold on the test distribution and report it as a valid tuned procedure. Report the sensitivity curve; selection, if needed, must use independent validation data.

## Priority 5 — Then evaluate the generator, not before

Once decoder behavior is understood, repeat the same conditional `x` grid with generated coordinates from the trained conditional DDPM. Compare against (i) X-ignored control, (ii) conditional empirical resampling, (iii) projected ambient conditional DDPM, and (iv) the already implemented but not-yet-run **label-aware branch/radius baseline**. The label-aware baseline is especially important for assessing whether specialized modeling of center/arm labels and conditional radii is a simpler solution.

At each `x` report conditional intrinsic W2, center probability error, arm probability errors, sampling validity, and runtime. Keep independent evaluation draws and matched train/test splits. The 40-update/12-step single-seed run only confirms the code pathway; it cannot establish success or failure of conditioning. Expand training only after the oracle-coordinate diagnostic is reproducible.

## Suggested small output package

Please send **(1)** a table/CSV with one row per `scenario x x x bandwidth x reference seed`, including center TP/FN/FP/TN, intrinsic reconstruction MSE and the `g_j(E(v))` values; **(2)** two plots, center mass vs bandwidth for S1 and false center mass vs bandwidth for S2, raw and thresholded; **(3)** one reference plot with arm counts/radial distribution; **(4)** the run configuration, code commit and test log.

**What would count as progress?** We can explain *where* the center atoms go and *which* choices fix or worsen the issue. A low MSE alone is not a pass criterion. No model-superiority or general statistical consistency claim is required for this debugging stage. If balanced references or better bandwidth restore the true-center reconstruction, the existing method may be adequate; if they do not, discuss a new decoder or calibration mechanism with Yidong and Changjun before scaling training.

---

*Technical note for the theory coauthor (not required for implementation):* The tripod center rule is `g_j=2 T_j-T`. Strictly negative `max_j g_j(E(v))` gives a stable vertex output. This explains why reference imbalance can cause atom loss even with exact encoded inputs. A vertex atom and substantial continuous near-vertex mass are different targets; W2/MSE alone cannot identify which one is being recovered.
