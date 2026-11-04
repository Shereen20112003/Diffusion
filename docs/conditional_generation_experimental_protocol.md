# Conditional Generation of Metric-Valued Data: Experimental Protocol

Prepared: 2026-11-01.

**2026-11-03 update:** This earlier multi-geometry proposal is retained as a
historical design document. The separately approved next step is the
[conditional metric-star pilot](conditional_metric_star_pilot.md), implemented
with a bounded single-seed smoke run. That later protocol supersedes the
execution order and star design below for the current task; it does not
authorize the other datasets, a full study, or external baseline installation.

**Status: proposed design only. No implementation, training, baseline
installation, or resumption of stopped experiments is authorized.**
Dataset choices and budgets below are recommendations for approval before
execution, not decisions attributed to the collaborators.

## 1. Purpose and relationship to the supplied report

Yidong requested a review of diffusion precedents and baselines to compare
later. Section 8, pages 9-10, of *Conditional Generation of Non-Euclidean
Data: Diffusion Models and Flow Matching* additionally recommends paired
data, held-out conditions, matched comparisons, conditional distributional
scores, and calibration. The [literature assessment](conditional_generation_of_non_euclidean_data_literature_review.md)
records the source checks and qualifications.

This protocol proposes specific simulations implementing those principles.
The data-generating equations below are our proposed designs, not equations
copied from the PDF. We retain conditional DDPM as the primary method;
the PDF's discussion of FrechetFlow does not authorize a switch to flow
matching. A flow comparator isolates the generator objective.

The existing two-moons and eight-Gaussians results concern unconditional
generation. They are separate evidence and will not populate conditional
result tables. A conditional model learns the entire law of Y given X=x,
not just a conditional Frechet mean or pooled marginal distribution.

## 2. Questions and staged scope

| Question | Experiment or control |
|---|---|
| Does the fitted model use X and recover conditional variability? | Known-law simulations, fixed held-out conditions, event probabilities, and a model trained without X. |
| Does diffusion improve over local reuse of training objects? | Kernel-weighted conditional resampling and nearest-neighbor resampling. |
| Does the coordinate representation/decoder help? | Direct conditional DDPM versus coordinate DDPM, with repair and reconstruction errors reported separately. |
| Does the diffusion objective help rather than just the decoder? | Conditional flow matching using exactly the same fitted encoder and decoder. |
| Is the framework useful beyond smooth manifold points? | Points on a fixed branching metric star, not generated graph topologies. |
| Can it learn a law over random distributions? | Measure-valued responses, evaluating between-object variation rather than pooled atoms. |

Recommended order: measure-valued known-law pilot; fixed-star pilot;
five-seed matched studies after approval; specialized external comparison;
finite-observation extension. These stages are not all mandatory to answer
the initial literature request. SPD and sphere benchmarks are optional
separate extensions, not prerequisites for the first pilot.

## 3. Paired synthetic data with known conditional laws

### 3.1 Primary study: one-dimensional random probability measures

Let X be uniform on [-1,1]. At each x, draw a component C in {1,2,3} with
probabilities softmax(-2x, 0, 2x). Independently draw U from Uniform(-0.15,0.15),
V from Uniform(-0.03,0.03), and H from Uniform(0,0.65). Define

$$
\ell=b_C+0.4x+U,\qquad s=a_C(1+0.2x)+V,
\quad b=(-1,0,1),\quad a=(0.25,0.4,0.55).
$$

For K=32, set u_j=(j-1/2)/K and

$$
q_j=\ell+s\{(1-H)\Phi^{-1}(u_j)+H[-\log(1-u_j)-1]\},
\qquad Y=\frac1K\sum_{j=1}^K\delta_{q_j}.
$$

The positive scale and increasing shape functions produce sorted atoms.
All randomness is bounded at the fixed quantile grid. Each **entire
32-atom measure is one response**, and repeated draws at the same x
produce distinct measures. X changes mixture weights, location, and scale.
The simulator supplies reference samples from the known conditional law;
its parameters are never given to the fitted models.

Use the exact inner metric

$$
d(Y,Y')^2=W_2(Y,Y')^2=\frac1K\sum_j(q_j-q'_j)^2.
$$

This is a finite-dimensional, geometrically simple quantile benchmark.
It is suitable for a controlled comparison, not evidence of performance
on arbitrary high-dimensional populations or an advantage over all
Euclidean representations.

### 3.2 Complementary study: points on a fixed three-edge star

Keep three unit-length edges joined at one vertex, with the existing
one-hot/radius representation. For X uniform on [-1,1], use

$$
p_0(x)=0.10+0.05x,\qquad
(p_1,p_2,p_3)=(1-p_0(x))\operatorname{softmax}(-2x,0,2x).
$$

With probability p_0 return the vertex. Otherwise choose branch B with
the normalized branch probabilities and draw

$$
R=0.2+0.8Z,\qquad Z\sim\operatorname{Beta}(2+x,2-x),
\qquad Y=R e_B.
$$

The metric is |r-r'| on the same branch and r+r' on different branches;
distance to the vertex is r. The exact existing folded-moment Frechet
decoder applies to this fixed tree. The conditional branch, vertex, and
radius laws are known. The absence of nonvertex training mass below 0.2
makes spurious near-vertex mass diagnostically visible.

A whole-graph generator changes the wrong random object. Embedding the
star in Euclidean coordinates does not make its branching point a smooth
manifold or preserve the required intrinsic metric.

### 3.3 Later extension: finite observations within a random measure

Only after the exact-object study, observe each latent measure through
m=32,128,512 independent atom draws. Convert each observed empirical
measure to K=32 midpoint empirical quantiles with a fixed, documented
inverse-CDF convention. Train using only observed representations;
retain latent measures solely for simulator-based evaluation.

Preserve latent-object IDs: all observations of the same object belong
to one split. Keep n, the number of paired objects, distinct from m,
the number of observations within an object. Report separately the
known-law error for latent objects, error against the observed-object
law, and empirical-measure/quantile-discretization error. An oracle
exact-object arm is a labeled sensitivity reference, not extra training
information for the noisy arm.

This extension does not automatically reproduce PerturbDiff: its
control-population and perturbation inputs must be matched in a separate
task before claiming a comparison.

## 4. Proposed methods and baseline eligibility

| Method | Definition and role |
|---|---|
| Coordinate conditional DDPM + learned Frechet decoder | Primary proposed method. Learn the law of E(Y) given X; reconstruct each generated coordinate with positive reference weights. |
| Coordinate conditional DDPM + kernel Frechet decoder | Shares generated coordinates with the primary decoder; isolates learning of reconstruction weights. |
| Direct conditional DDPM | Generate three ambient coordinates for the star or 32 atom coordinates for measures. Apply the documented star projection or increasing rearrangement; save raw outputs and repair displacement. |
| Same-map conditional flow matching | Use exactly the primary fitted E and learned decoder. Train a conditional coordinate velocity field on straight Gaussian-to-encoded-data paths, with target z_data-z_noise. This is our control implementation, not an official WFM reproduction. |
| Kernel conditional resampling | Sample whole training responses with weights proportional to exp(-(x-X_i)^2/(2h^2)); choose h from {0.05,0.1,0.2} using validation conditional W2. Normalize stably and report effective sample size. |
| Nearest-neighbor conditional resampling | Uniformly sample responses from k nearest training covariates; choose k from {32,64,128} on the same validation metric. |
| X-omitted coordinate DDPM | Same primary representation/decoder and training budget, but no covariate input. Tests whether pooling can conceal failed conditioning. |
| Real-real reference | Independent samples from the true law at each x, with the same sample counts. Finite-sample context, not a strict lower bound. |

Fit the representation and reference objects on training Y only. Use eight
FPS encoding landmarks plus a base, and 64 FPS reconstruction objects.
Fit the learned decoder once per dataset/seed; share its exact frozen
parameters across coordinate DDPM, coordinate flow, and X-omitted controls.
Its reconstruction objective does not use X. Tune kernel/learned decoder
bandwidth over {0.1,0.3,1.0} using fit-validation raw embedded MSE, separately
from the resampling bandwidth h. Do not jointly retune a decoder for the
preferred generator.

For DDPM, add x to the denoiser input at every noise step. Do not noise X,
infer it from a generated sample, or silently accept a missing condition.
Fit any X normalization on training pairs. Use identical conditioning
features for the direct and coordinate generators: x and four fixed
sine/cosine bands. Retain four spatial bands, width 128, batch 128, and
Adam at 0.001 for the first proposed study. The coordinate flow uses the
same covariate features and hidden widths, with continuous-time features
rather than a DDPM step index. Count parameters rather than claiming
identical parameter counts across different output dimensions.

### Specialized comparison gate

The primary external candidate for measure-valued outputs is
[Wasserstein Flow Matching](https://proceedings.mlr.press/v267/haviv25a.html).
Before running it, inspect the actual conditional code, fix a source
revision, and specify how scalar x and 32-atom measures enter it.
A 1D quantile-based specialization is a labeled adaptation, not an
official reproduction. The abstract-level GWFM check informs the
nested-law comparison but does not establish a ready conditional runner.

[PerturbDiff](https://arxiv.org/abs/2602.19685) is a close diffusion precedent,
but a synthetic scalar-X study lacks its control-population interface.
Do not pass the unknown target population as a condition. Include it only
after matching the observation/conditioning regime. Likewise, distribution-
conditioned transport supplied a target-distribution embedding is not a
fair predictor-only baseline.

If SPD responses are selected instead, prioritize
[SPD-DDPM](https://arxiv.org/abs/2312.08200) and
[DiffeoCFM](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5616112a0120c15bf7d47a6bccc21bc3-Abstract-Conference.html),
after agreeing on a common matrix metric. No specialized plug-in
fixed-star-point diffusion baseline was established by this review.
Do not substitute DeFoG, BWFlow, or a phylogenetic-tree generator.

Internal pilots may proceed without an external reproduction after
authorization, but remain incomplete comparative evidence until the
relevant specialized comparator or an explicit scope limitation is reported.

## 5. Splits, evaluation conditions, and proposed budgets

| Item | Small pilot, proposed only | Five-seed study, proposed only |
|---|---|---|
| Independent data/training seeds | 2000 | 2000-2004, separate namespace from old runs |
| Paired training / fit-validation objects | 2048 / 512 | 4096 / 1024 |
| Generator updates | 1000 | 10000 |
| Learned decoder updates per bandwidth | 1000 | 1000 |
| DDPM reverse steps | 200 | 200 |
| Flow sampling | 200 fixed Euler steps; report 200 velocity evaluations | Same |
| Test samples per condition, per draw | 128 generated + 128 reference | 512 generated + 512 reference |
| Independent evaluation draws | 2 | 3 |
| Test covariates | {-0.8,-0.4,0,0.4,0.8} | -0.9+0.1j for j=0,...,18 |
| Conditional-validation covariates | {-0.9,-0.5,-0.1,0.3,0.7} | -0.95+0.1j for j=0,...,19 |

Training and fit-validation X values are independently continuous uniform
draws. Fixed evaluation conditions and their responses are excluded from
fitting. These are **held-out-condition interpolation** studies, not
extrapolation or deliberately withheld covariate intervals. A future
gap/extrapolation study requires separately declared training support.

Use separate deterministic RNG streams for training data, validation
data, initialization, optimizer draws, conditional validation, generated
samples, and test reference samples. All methods share paired splits and
test references within a seed, without sharing generated outputs except
the deliberate decoder comparison.

Pilot checkpoint candidates are {500,1000}; full-study candidates are
{2500,5000,10000}. Select each neural generator using mean squared
conditional object-space W2 on the conditional-validation grid, with 128
generated/reference objects and one frozen validation draw per grid point.
Use the learned decoder for primary coordinate-model selection; evaluate
its kernel-decoder arm using the same selected coordinate generator.
Resampling selects its three candidates on the same grid and score.
Ties select the earlier checkpoint or first listed candidate.
Test outcomes never select a checkpoint, seed, bandwidth, or method.

Equal updates and equal sampling steps do not equal total compute. Report
parameter count, tuning candidates, gradient updates, network evaluations,
decoder cost, training/selection/sampling wall time, and peak memory.
Use two CPU threads initially; do not promise a wall-clock duration before
a bounded pilot measures it. External methods require separate approved
resource budgets and isolated environments with local-only logging.

## 6. Primary metric and diagnostics

At each held-out x_j, compute exact equal-mass empirical W2 between the
generated and true object samples using linear assignment on squared
object distances. For measure responses, this is an **outer** W2 whose
ground cost is **inner** 1D W2 squared, not Wasserstein distance between
all pooled atoms.

For each seed s and draw r, define

$$
R_{s,r}=\left[\frac1J\sum_{j=1}^J
  \widehat W_{2,d}(\widehat P_{x_j},P_{x_j})^2\right]^{1/2}.
$$

Average R over draws within each seed, then report the mean and sample
standard deviation across five independent seeds. Also preserve every
condition/draw score and report paired seed differences versus each
prespecified baseline. The finite uniform grid approximates a
uniform-covariate risk; it is not an exact population integral.
SD is not a confidence interval, and three evaluation draws are not
three independent training runs.

| Diagnostic | Prespecified measurement |
|---|---|
| Star probabilities | Vertex and each branch probability error at every x; half the sum of absolute errors for the four-category law. Vertex uses radius <=1e-6; report spurious mass with radius in (1e-6,0.2). |
| Star geometry | Conditional radial distributions and branch-radius joint fit, plus valid-object fraction and projection displacement. Flag branch-specific scores as unavailable if sample counts are insufficient, never zero. |
| Measure events | P(mean(Y)<=0), P(mean(Y)<=-0.5), and P(sd(Y)>=0.5) as functions of x; mean and sd are properties of each whole measure. |
| Event probability fidelity | Generated event fractions versus true probabilities: analytic star probabilities; independent simulator reference samples for measure events. Report Monte Carlo uncertainty. |
| Predictive calibration | Brier scores on fresh paired test observations and reliability plots in ten fixed [0,1] probability bins. Generate predictive event probabilities at each observation's X, not the nearest test grid point. Empty bins remain empty. |
| Diversity | Conditional variance of measure means/scales or star radii, and pairwise object distances. Compare to the true law, not merely to zero diversity. |
| Representation and decoding | Held-out object reconstruction error, nearest-reference distance, localization and weight-sum error. Keep these separate from conditional generative error. |
| Object validity | Finite sorted atoms or valid star points; show raw direct-DDPM violations before repair. Invalid objects are not silently discarded. |
| Conditional dependence | Compare with the X-omitted model. Report conditional scores even if pooled marginal W2 is favorable. |

For event-probability reference curves, use 10000 independently simulated
objects per condition in the full study (2000 in a pilot), shared across
methods. These are diagnostic references, not a replacement for the
equal-sized W2 references. For Brier/reliability evaluation, use 512 fresh
paired observations per seed in the full study (128 in a pilot) and 256
generated objects at each observation's X (128 in a pilot). Account for
this sampling cost separately. These reference/calibration samples never
enter selection. Report event-specific results and bin counts, not a
single calibration number without sample sizes.

No fixed success threshold or statistical significance claim is asserted
in advance. Show failures and unfavorable baselines. In particular, a
small conditional W2 does not by itself establish correct rare-event
probabilities, a theorem, or novelty.

## 7. Implementation boundary and readiness criteria

Current reusable components are `ObjectSpace`, `MetricEncoder`,
`FrechetDecoder`, and `object_w2` in `experiments\metric_spaces.py`, plus
the DDPM schedule and noise helpers in `experiments\run_experiments.py`.
The existing `Denoiser.forward(coordinates, step)` and fit/sample
interfaces do not accept X. Existing samplers generate only unconditional
objects. A future conditional runner must supply explicit paired data,
conditions, conditional validation, and method/version metadata rather
than relabeling an existing runner.

Before scientific training, implement and check:

1. Exact tree/quantile metric cases, sorted-atom validity, normalized
   conditional probabilities, distinct x-dependent simulator laws, and
   exact Frechet reconstructions on hand-checkable examples.
2. Condition-shape/range validation and a sampler contract preserving the
   supplied x at every step; an X-omitted control must be explicitly named.
3. Split/RNG isolation, including object IDs in the finite-observation
   extension, and no access to test arrays during selection.
4. Outer-W2 assignment against small hand-computed costs, correct
   condition/draw/seed aggregation, probability diagnostics, and explicit
   reporting of invalid outputs and undefined metrics.
5. A tiny end-to-end smoke run only after execution approval. It is not
   scientific evidence and must use a separate output directory.
6. One bounded pilot with an agreed runtime ceiling, followed by an
   explicit decision on the five-seed and external-baseline budgets.

Save immutable configuration/source hashes, source revisions for external
methods, paired data IDs, fitted references, selected checkpoints, raw and
decoded samples, conditional metrics, timing, and per-unit completion
markers. Resume only compatible units. Never overwrite the original
unconditional study or silently select only successful seeds.

Planned reporting consists of a method-by-dataset risk table, per-condition
risk curves, event-probability and calibration plots, fixed-condition
sample panels, per-seed scores, and an explicit completion matrix.
No conditional result table is currently populated.

## 8. Interpretation and theoretical limits

Adding X to a network is established prior art. The report identifies
candidate contributions, not verified gaps. A useful result would need
to demonstrate a specific benefit or establish a nontrivial guarantee
for the stated response space and observation model.

The [Tang-Lin-Yang diffusion study](https://arxiv.org/abs/2409.20124) and
[Tang-Yang minimax study](https://arxiv.org/abs/2506.07504) already address
conditional-law estimation under smooth embedded-manifold assumptions.
Their theory is not automatically a general-metric W2 bound for our
trained decoder. [Conditional Wasserstein distances](https://jmlr.org/papers/v26/24-0586.html)
also caution against replacing conditional error with joint-law transport.
Applying the current unconditional recovery bound pointwise in x alone
does not solve conditional learning, reference coverage, or observation
noise. These simulations cannot establish asymptotic rates with fixed
sample/reference budgets.

The review and this protocol are the deliverables completed now.
Conditional implementation, comparisons, theory, and training remain
future work requiring explicit approval.
