# Conditional generation for non-Euclidean data

Review recorded: 2026-11-01.

## Assessment and scope

The review supports investigating conditional generation, but does not establish
novelty. Conditional generation is already studied for rotations, SPD matrices,
molecular structures, graphs, and function-valued outputs. The stronger question
is whether our general-metric-space formulation, reconstruction mechanism, and
conditional recovery guarantees offer something those methods do not.

Here, conditional generation means generating an output given external
information, rather than merely conditioning a denoising loss on the original
sample. The review covers diffusion, normalizing flows, conditional density
estimation, related recovery theory, and metric-space reconstruction.

The sources below were checked through primary abstracts, proceedings, and
selected full-text sections. This is not an exhaustive citation census, a
replication of the methods, or an audit of every proof. No exact prior match to
our complete proposed construction was established among the verified sources;
that is not proof that none exists.

## Existing conditional approaches

| Paper | Conditioning and generated output | Relevance |
|---|---|---|
| [SPD-DDPM](https://arxiv.org/abs/2312.08200), AAAI 2024 | Covariates to SPD matrices; explicitly learns a conditional distribution and uses generated samples to obtain a Frechet-mean prediction. | A close application-level comparison connecting conditional diffusion, non-Euclidean responses, and Frechet means. Relies on SPD-specific operations and architecture. |
| [Unified framework for diffusion generative models in SO(3)](https://arxiv.org/abs/2312.11707), AAAI 2024 | Images to object rotations; tidal-field information to galaxy orientations. | Genuine covariate-conditioned manifold diffusion using SO(3)-specific heat-kernel structure. |
| [DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking](https://arxiv.org/abs/2210.01776), ICLR 2023 | Protein and ligand information to plausible ligand binding poses. | Conditional generation on a product of translation, rotation, and torsional degrees of freedom. Geometry and representation are task-specific. |
| [De novo design of protein structure and function with RFdiffusion](https://pmc.ncbi.nlm.nih.gov/articles/PMC10468394/), Nature 2023 | Motifs, binding targets, and structural constraints to protein backbones. | Established conditional generation of geometrically structured objects, using protein-specific representations and constraints. |
| [GraphGUIDE](https://arxiv.org/abs/2302.03790), 2023 preprint | Structural conditions to graphs through discrete Bernoulli diffusion. | Relevant when each observation is an entire network, not when generating a point on a fixed metric tree or graph. |
| [Practical and Asymptotically Exact Conditional Sampling in Diffusion Models](https://arxiv.org/abs/2306.17775), NeurIPS 2023 | The Twisted Diffusion Sampler conditions an existing diffusion model using sequential Monte Carlo; includes a Riemannian protein-design application. | Alternative to training a separate conditional model. Asymptotic sampling correctness concerns the model-defined conditional target, not automatically the true data distribution. |
| [HuManiFlow](https://arxiv.org/abs/2305.06968), CVPR 2023 | Image to distributions of human poses and shapes, with rotations modeled on SO(3). | Conditional normalizing flows representing multiple plausible outputs; a non-diffusion comparison. |
| [Delving Into Discrete Normalizing Flows on SO(3) Manifold for Probabilistic Rotation Modeling](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_Delving_Into_Discrete_Normalizing_Flows_on_SO3_Manifold_for_Probabilistic_CVPR_2023_paper.html), CVPR 2023 | Input observations to rotation distributions. | Further evidence that the comparison must include conditional flows, not only diffusion. |
| [Implicit-PDF](https://proceedings.mlr.press/v139/murphy21a.html), ICML 2021 | Image and candidate rotation to a conditional density on SO(3). | Earlier full-distribution modeling of manifold-valued responses; a density estimator rather than a diffusion sampler. |
| [Guided Diffusion Sampling on Function Spaces with Applications to PDEs](https://arxiv.org/abs/2505.17004), NeurIPS 2025 | Sparse or noisy measurements to function-valued solutions in PDE inverse problems. | Uses function-space diffusion and observation guidance, with Banach-space analysis. Not a framework requiring only arbitrary metric-space distances. |

### Why SPD-DDPM matters particularly

Its conditional section explicitly generates matrices given covariates and then
computes their Riemannian center. However, its Frechet mean summarizes
already-generated samples. Our proposed Frechet decoder reconstructs each
generated object from reference objects and generated coordinates. These are
related but different uses.

The conditional mechanism is verified in the full-text section "Conditional
Symmetric Positive Definite Denoising Diffusion Probabilistic Model," including
its covariate-dependent denoiser and conditional training/inference algorithms.
Its existence as relevant prior work does not imply that all its derivations
have been independently validated.

## Closest theoretical comparison

[Conditional Diffusion Models are Minimax-Optimal and Manifold-Adaptive for
Conditional Distribution Estimation](https://arxiv.org/abs/2409.20124),
Tang, Lin, and Yang, ICLR 2025, is essential reading.

It studies estimation of the conditional law from paired observations:

$$
(X_i,Y_i)_{i=1}^n \longrightarrow \widehat P_{Y\mid X=x}.
$$

It establishes classical conditional-density results and manifold-adaptive
guarantees. In its manifold setting, the risk is the average conditional
1-Wasserstein error:

$$
\mathbb E_X[W_1(\widehat P_{Y\mid X},P_{Y\mid X})].
$$

The rate depends on intrinsic dimensions and smoothness parameters. The
response manifold can vary with the covariate under regularity assumptions.
The inspected theorem is Theorem 2 and its assumptions in the arXiv v1 text.

The main distinction from our intended setting is that variables have ambient
Euclidean coordinates and diffusion operates in that ambient space. Response
supports satisfy smooth-submanifold and conditional-density assumptions,
including regular local representations and a density lower bound. This is not
the same as a distance-based framework for general metric spaces with possible
branching or nonsmooth geometry. The covariate complexity is described through
an intrinsic packing condition.

We cannot claim that existing work lacks conditional distribution-recovery
theory for manifold-supported data. Nor should a W1 guarantee be described as
the same result as our intended object-space W2 recovery bound.

[Wasserstein Generative Learning of Conditional
Distribution](https://arxiv.org/abs/2112.10039) is also relevant. It develops
conditional generators and nonasymptotic analysis. Its main formulation uses
vector-valued predictors and responses. "Wasserstein" in its title does not
mean each response is itself a probability distribution.

## Related architecture and output-space work

### Global Frechet Manifold Learning for Random Objects

[Gajardo and Muller, JMLR 2026](https://www.jmlr.org/papers/v27/23-1392.html)
use distance-based manifold learning to obtain Euclidean representations, then
global Frechet regression to map representations back to metric-space objects.
The paper includes distribution-valued data.

Embedding objects in Euclidean space and reconstructing through Frechet
regression is therefore not, by itself, our novelty. We must explain what the
distance-contrast construction, generative model, and recovery analysis add.

### E2M

[End-to-End Deep Learning for Predicting Metric Space-Valued
Outputs](https://jmlr.org/papers/v27/25-2364.html), JMLR 2026, predicts objects
through input-dependent weighted Frechet means, with approximation and
convergence theory. It is not itself a full conditional-law sampler.

Our distinction would be generating a distribution of plausible objects rather
than returning one prediction. The reconstruction mechanism still needs to be
situated relative to E2M.

### Wasserstein Flow Matching

[Wasserstein Flow Matching: Generative Modeling Over Families of
Distributions](https://proceedings.mlr.press/v267/haviv25a.html), ICML 2025,
generates observations that are themselves distributions, represented
analytically or by point clouds. It is important for our proposed
distribution-valued experiments.

Its phrase "conditional flows" refers to the flow-matching construction; that
phrase alone does not establish an external-covariate conditional-generation
experiment. This review classifies it as relevant distribution-valued generation,
not verified evidence of our exact conditional task.

### Flow Matching on General Geometries

[Chen and Lipman, ICLR 2024](https://arxiv.org/abs/2302.03660) provide a foundational
geometric-generation comparison. Broad Riemannian settings are different from
an arbitrary metric-space formulation without smooth geometric operations.

## Candidate positioning, not an established novelty claim

> Conditional distribution learning for metric-space-valued responses using
> distance-based coordinates, Euclidean conditional diffusion, and weighted
> Frechet reconstruction, with an object-space recovery analysis separating
> coordinate-generation, reference-coverage, localization, and numerical
> reconstruction errors.

| Possible distinction | Evidence required |
|---|---|
| Broader output geometry | Explicit assumptions and a meaningful example beyond smooth manifolds, not merely a sphere demonstration. |
| Reduced geometric requirements | State required operations: distances and a reconstruction solver rather than tangent-space scores or heat kernels. A distance encoding alone does not solve reconstruction. |
| Conditional-law recovery | Compare distributions at the same covariate, not only marginal sample quality. |
| Reconstruction analysis | Control generated coordinates, including those outside the observed coordinate image. |
| Practical value | Conditional fidelity, multimodality, valid outputs, and competitive cost against appropriate alternatives. |

The existing manuscript theorem being conditional on the fitted model does not
mean it proves generation of a response conditional on a covariate.

Applying the current recovery inequality separately at each covariate might
yield a useful conditional bound under suitable common assumptions. That alone
may be a relatively straightforward extension. Stronger theoretical work would
control conditional coordinate-learning error, support/reference coverage, and
reconstruction error to obtain a meaningful statistical guarantee.

The current direct-MLP pilot does not automatically inherit the weighted-Frechet
theorem. Conditional generation and a conditional theorem have not been
implemented or established in this repository.

## Experimental implications

The following are proposed additions, not completed experiments:

- A conditional toy study in which a covariate changes mixture weights or
  locations. Evaluate at fixed held-out covariates, not just pooled samples.
- An established geometric benchmark, such as SPD-valued responses, with a
  relevant specialized conditional baseline.
- A broader metric-space example, such as points on a branching metric graph,
  with the output space defined precisely.
- Simple conditional baselines: nearest-neighbor or kernel-weighted resampling
  of training outputs, plus an appropriate coordinate-space conditional
  generator. Test whether diffusion adds value beyond local empirical sampling.
- Conditional distribution error where ground truth is available, mode
  probabilities, validity, diversity, and repeated-seed uncertainty. Good
  marginal W2 alone can hide a model that ignores its conditioning input.

Methods, metrics, and baselines must be valid for the selected object geometry.
A whole graph, a point on a fixed metric graph, and a phylogenetic tree are
different response spaces. Likewise, Wasserstein evaluation of ordinary samples
is different from generating objects that are probability measures.

## Suggested collaborator update

> I broadened the review to include conditional diffusion, normalizing flows,
> and conditional density estimation for non-Euclidean outputs. There are
> established examples on SO(3), SPD matrices, molecular structures, and graphs.
> Two particularly relevant papers are SPD-DDPM, which explicitly models
> conditional SPD-valued distributions, and Tang-Lin-Yang's ICLR 2025 paper on
> conditional diffusion with manifold-adaptive statistical guarantees. There is
> also prior work on Euclidean representations with Frechet reconstruction.
> The conditional direction still looks worth pursuing, but the potential
> contribution should be framed around our specific general-metric-space
> setting, reconstruction mechanism, and conditional recovery guarantees rather
> than conditioning alone. We should compare these assumptions carefully and
> demonstrate conditional distribution recovery on a setting where the broader
> metric formulation matters.
