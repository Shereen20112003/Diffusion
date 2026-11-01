# Conditional Generation of Non-Euclidean Data: Diffusion Models and Flow Matching

Literature Review and Contribution Assessment

Review recorded: 2026-11-01.

## Assessment and scope

The review supports investigating conditional generation, but does not establish
novelty. Conditional generation is already studied for rotations, SPD matrices,
molecular structures, graphs, functions, and distribution-valued outputs. The stronger question
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

## Diffusion Precedents and Baseline Selection

A [shareable PNG summary](figures/diffusion_precedents_and_baseline_selection.png)
separates selected literature precedents from experimental completion.

This section addresses the collaborator's request to inspect the supplied
PDF, focus on diffusion precedents, and identify comparisons to run later.
It is a literature and implementation-availability assessment, not an
experimental reproduction.

The source-checking pass now covers **all 42 works: 9 with relevant
full-text inspection and 33 at primary abstract/metadata level**. There are
no unchecked bibliographic reading leads in the register. This completes
the bounded source-checking pass, not a full-text review of every paper,
an exhaustive novelty search, or verification of every claim in the PDF.
Abstract checks establish the stated construction, not validity of every
proof or reported result.

The [Conditional Generation Experimental Protocol](conditional_generation_experimental_protocol.md)
turns the report's Section 8 recommendations into proposed paired-data
studies, baseline controls, conditional metrics, and staged budgets.
It is a design document only; implementation and training are not authorized
by this review.

| Diffusion precedent | What the paper establishes | Implication for our comparison |
|---|---|---|
| [SPD-DDPM](https://arxiv.org/abs/2312.08200), AAAI 2024 | External predictors enter an SPD-valued denoiser; generated matrices are summarized by a Riemannian center. | Priority conditional-diffusion baseline if SPD responses are selected. Full conditional-law evaluation should supplement its mean-prediction evaluation. |
| [PerturbDiff](https://arxiv.org/abs/2602.19685), 2026 preprint | A population distribution is a random output, conditioned on a control population and context through an embedding-based diffusion formulation. | Closest diffusion comparison for a conditional distribution-valued study. Match the outer-law target and finite-batch observation model, not only pooled cells. |
| [Riemannian Score-Based Generative Modelling](https://arxiv.org/abs/2202.02763), NeurIPS 2022 | Intrinsic manifold noising and generation, with sphere experiments. | First intrinsic diffusion candidate for our current sphere-valued data. Its existing sphere example is not evidence that our covariate-conditioned task has been implemented. |
| [Riemannian Diffusion Models](https://arxiv.org/abs/2208.07949), NeurIPS 2022 | Riemannian diffusion with geometry-aware score/likelihood constructions. | Alternative intrinsic sphere baseline; not necessary to reproduce both foundational methods before a first comparison. |
| [Point Set Diffusion](https://arxiv.org/abs/2410.22493), ICLR 2025 | Point-configuration generation on general metric spaces, plus conditioning on observed points through masks. | Important for the broad metric-space positioning. Not ordinary external-covariate regression, and not automatically a generator of single points on a fixed star. |
| [Michelangelo](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ea1a7f7bc0fc14142106a84c94c826d0-Abstract-Conference.html), NeurIPS 2023 | Image/text-conditioned latent diffusion followed by a shape decoder. | Direct precedent for conditional representation-and-decoding pipelines; a numerical baseline only if a matching shape task is chosen. |
| [Twisted Diffusion Sampler](https://arxiv.org/abs/2306.17775), NeurIPS 2023 | Conditional sampling from a learned prior via sequential Monte Carlo. | A different conditioning mechanism, suitable only when the observation/likelihood interface and target match. |

Other diffusion precedents in the PDF remain relevant citations:
financial correlation-matrix DDPM; image-conditioned SE(3) pose diffusion;
torsional diffusion, DiffDock, RFdiffusion, and equivariant molecular
diffusion; MatterGen; DiGress and phylogenetic discrete diffusion;
Geometric Neural Diffusion Processes; and Infinite Brush. Their
application-specific outputs do not make all of them mandatory numerical
baselines for our current sphere, star, and measure-valued examples.

**Point Set Diffusion requires particular attention.** Sections 2.1 and 3
of its [full text](https://arxiv.org/html/2410.22493v1) explicitly use a
complete separable metric domain and point configurations, rather than only
smooth manifolds. Section 3.3 and Algorithm 1 condition on a binary spatial
or temporal mask; they do not establish arbitrary subject-covariate
conditional regression. Section 3.2 uses a Gaussian-mixture location model
for the Euclidean implementation, and Section 4 evaluates Euclidean
spatial/spatiotemporal datasets. Its existence rules out a blanket claim
that all diffusion precedents require smooth manifold structure, but its
generality should not be confused with a ready implementation for every
metric-valued response.

### Official implementation availability

The following author-linked repository documentation was inspected.
Availability and documented entry points are not installation or
reproduction guarantees; no commands from these repositories were run.

| Candidate | Official repository and observed support | Work needed before comparison |
|---|---|---|
| Riemannian Score-Based Generative Modelling | [oxcsml/riemannian-score-sde](https://github.com/oxcsml/riemannian-score-sde), commit `b35b739464ca8264e418e400301f5f45cb7a16b3`. README documents an `s2_toy` experiment, JAX, and a modified geomstats dependency. | Isolate/pin the older environment, use local logging, adapt the data to our three-component cap distribution, and evaluate with the same geodesic W2 and budgets. Do not substitute the paper's likelihood numbers for our metric. |
| SPD-DDPM | [li-yun-chen/SPD-DDPM](https://github.com/li-yun-chen/SPD-DDPM), commit `a6a65d13d80369baa8f2dc8eac352a14ee4a919e`. README documents conditional/unconditional code, predictor columns, and R scripts for Frechet-mean evaluation. | Select an actual paired SPD dataset, check the metric and conditional interface, and retain generated matrices for full-law evaluation rather than reporting only their mean. |
| PerturbDiff | [DeepGraphLearning/PerturbDiff](https://github.com/DeepGraphLearning/PerturbDiff). README documents a PyTorch runtime, configurations, and conditional sampling. | Align controls, covariates, population batches, and outer-law evaluation. Its released biological pipeline is not a drop-in replacement for our unconditional 32-atom example. |
| Wasserstein Flow Matching | [DoronHav/WassersteinFlowMatching](https://github.com/DoronHav/WassersteinFlowMatching), commit `9a90030ca739d892c2973bbcdfc9fbaced974154` for the inspected README. Documents JAX, Flax, JAX-OTT, point clouds, and Gaussians. | Retain as a non-diffusion geometric comparator. Match the individual measure representation, inner metric, and outer distributional evaluation. |

The minimum comparison set depends on the selected experiment: intrinsic
Riemannian diffusion for sphere-valued points; WFM for measure-valued
objects; SPD-DDPM plus DiffeoCFM for a future conditional SPD benchmark;
and PerturbDiff plus a conditional WFM construction for a future
conditional population benchmark. Conditional resampling and a generator
sharing our encoder/decoder are controls across the conditional settings.
There is no implication that all application families must be implemented.

For source transparency, the Nature article endpoint for MatterGen returned
HTTP 406. Its construction was checked through the
[author preprint](https://arxiv.org/abs/2312.03687), and the Nature title,
year, volume, and pages through Crossref DOI metadata. Point Set Diffusion's
available arXiv text is v1; an attempted v3 URL returned 404. Its ICLR 2025
publication was checked against the official proceedings. These access
details do not indicate that the methods were run.

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

**Correction after full-text inspection on 2026-11-01:** external conditioning
is verified. Sections 4.1.2 and 4.2.1 of the
[full text](https://arxiv.org/html/2411.00698v2) include organ/cell-state
conditioning and label-conditioned ModelNet40 generation, respectively.
The earlier version of this review left that capability unverified; that
classification is superseded. The phrase "conditional flows" alone would not
have sufficed, but these experiments provide the missing evidence.

The [author-linked repository](https://github.com/DoronHav/WassersteinFlowMatching)
is accessible; its README specifies JAX, Flax, and JAX-OTT. Repository
availability is not a successful installation or reproduction.

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

## Assessment of the collaborator's 2026-10-31 report

The supplied `non-euclidean-conditional-generation-2026-10-31.pdf` is a
14-page report retaining 42 works. Its accompanying evidence archive was not
supplied. We inspected the report and checked the central comparisons against
primary abstracts, selected full-text sections, and author-linked repositories.
This is not independent verification of every claim about all 42 works.

The report's discussion of "FrechetFlow" and velocity networks must not be
silently substituted for our current DDPM formulation. Its conclusions about
the statistical target, prior conditioning, reconstruction, and evaluation
are nevertheless applicable. It supports investigating a specific
general-metric conditional formulation, not asserting that conditioning or
non-Euclidean generation is new.

### Most consequential additions

| Precedent | Evidence checked | Consequence for this project |
|---|---|---|
| [PerturbDiff](https://arxiv.org/abs/2602.19685), 2026 preprint | Full text, Sections 4.1-4.5: the output is a random population distribution; Eq. (2) conditions on a control distribution, context, and perturbation. Eq. (6) includes those inputs in the denoiser. | A particularly close diffusion precedent for conditional distribution-valued responses and embedding-based generation. Must be discussed explicitly, not treated as ordinary single-cell generation. |
| [WFM](https://proceedings.mlr.press/v267/haviv25a.html), ICML 2025 | Full-text external-label experiments, as corrected above. | Both distribution-valued generation and its conditional version have precedents. Include a geometry-native flow comparator even if our method uses diffusion. |
| [DiffeoCFM](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5616112a0120c15bf7d47a6bccc21bc3-Abstract-Conference.html), NeurIPS 2025 | Primary abstract: matrix logarithm/normalized-Cholesky transformations and pullback metrics. | A close coordinate-generation comparator on matrices. The abstract alone does not establish every categorical/continuous conditioning interface claimed in the report. |
| [CorrFlow](https://arxiv.org/html/2610.13606v1), NeurReps 2026 workshop preprint | Full text, Section 2.2, Eqs. (3)-(4): regime indicator and continuous performance score enter the vector field. | Continuous external conditioning is already demonstrated in correlation coordinates. Its reported comparisons do not isolate a universal advantage of FM over diffusion. |
| [RWEFM](https://arxiv.org/abs/2609.25659), 2026 preprint | Primary abstract and formulation: distributions over measures on manifolds; geodesic-distance/projection requirements. | Relevant when an entire measure on a curved domain is one observation. Do not claim all geometric alternatives require closed-form heat kernels. The specific label/sequence interfaces in the report were not independently audited here. |
| [Conditional Wasserstein distances](https://jmlr.org/papers/v26/24-0586.html), JMLR 2025, and [COT-FM](https://proceedings.neurips.cc/paper_files/paper/2024/hash/aa1d8cf866c4a684ef2e066dd200f8e4-Abstract-Conference.html), NeurIPS 2024 | Primary abstracts: restricted conditional couplings; dynamic conditional OT, including infinite-dimensional settings. | A small joint Wasserstein distance is not automatically small average conditional Wasserstein error. Conditional theory comparisons must include flows, not only diffusion. |

PerturbDiff's Section 4.4 is especially important for a fair comparison:
finite cell batches instantiate embeddings, the implemented loss combines
MMD with pointwise MSE, and sampling uses DDIM in cell space. Its noise
construction uses a first-order approximation. We have not audited its proofs
or equated this implementation with exact infinite-dimensional diffusion.
The [official PyTorch repository](https://github.com/DeepGraphLearning/PerturbDiff)
is accessible and documents training and conditional sampling. Its README
records a code release, superseding the preprint abstract's prospective
"will be made publicly available" wording. No external baseline was installed
or run in this review.

### Baselines to compare, matched to our actual response spaces

**Completion status of the PDF's proposed experiments:** none of its
conditional comparative studies has been completed in this project.
Section 8 (pages 9-10) proposes a conditional evaluation programme; the
experiments described elsewhere in the report belong to the cited papers
and have not been reproduced here. Literature inspection and citation
integration are not experimental completion.

| Proposed comparison or evaluation in the PDF | Project status |
|---|---|
| Conditional generation from paired observations at a supplied covariate | Not implemented or evaluated. Existing studies are unconditional. |
| Matrix-valued conditional diffusion versus native geometric flow matching | Not run. SPD-DDPM, DiffeoCFM, and CorrFlow have not been reproduced. |
| Conditional distribution-valued generation against WFM/RWEFM or an embedding-based construction | Not run. WFM, RWEFM, and PerturbDiff have not been reproduced. |
| Conditional resampling and generators sharing the same encoder and decoder | Not run. Existing unconditional resampling and decoder comparisons do not satisfy this requirement. |
| Matched held-out covariates, conditional distributional scores, and event calibration | Not evaluated. Existing marginal W2 results are not conditional-law evaluations. |
| Variation among generated distributions, distinguished from within-object sampling noise | Not evaluated. The full unconditional distribution-valued study is also incomplete. |

The completed five-seed two-moons and eight-Gaussian studies belong to
the earlier manuscript protocol, not to this proposed conditional programme.
One sphere seed is complete; the full fixed-star and distribution-valued
studies remain incomplete.

| Setting | Comparison to prioritize | Scope and adaptation |
|---|---|---|
| Two moons / eight Gaussians | Existing direct DDPM and empirical training-object resampling; retain all landmark/decoder comparisons. | Already present in the completed five-seed planar runs. An eventual same-encoder/same-decoder generator comparison would isolate the diffusion objective rather than change the entire pipeline. |
| Points on the sphere | [Riemannian Score-Based Generative Modelling](https://arxiv.org/abs/2202.02763) or [Riemannian Diffusion Models](https://arxiv.org/abs/2208.07949); RFM as a non-diffusion alternative. | Highest-priority missing intrinsic baseline. Match the cap distribution, geodesic evaluation metric, data splits, and budgets. Current ambient DDPM plus normalization is not this baseline. |
| One-dimensional, 32-atom distribution-valued observations | WFM as a geometry-native flow comparator; retain sorted-atom DDPM and resampling. | Match the outer distribution of measure-valued objects, not only pooled atoms. Adapt the representation and cost to our benchmark; WFM's published application scores are not comparable to our toy W2. |
| Future conditional distribution-valued observations | PerturbDiff and a conditional WFM construction. | Define paired conditions and repeated distribution-valued outputs first. PerturbDiff's control-population/covariate interface is not a drop-in replacement for an unconditional sorted-atom runner. Separate between-object variation from finite within-object sampling noise. |
| Points on a fixed metric star | Current projected ambient DDPM and empirical resampling; explicitly model/evaluate branch probabilities, radii, and vertex mass. | DiGress, phylogenetic discrete diffusion, and PhylaFlow generate entire graphs/trees. They are citations for nearby structured spaces, not valid direct baselines for this response space. No specialized plug-in star diffusion baseline was established in this review. |
| SPD matrices, if later added | SPD-DDPM; DiffeoCFM as a flow comparison. | Not part of the current Section 6 execution. Define the SPD metric and conditioning variables before adapting these methods. |
| Correlation matrices, if later added | Conditional financial DDPM, DiffeoCFM, and CorrFlow. | Correlation constraints differ from unrestricted SPD matrices. Separate valid-object generation from post-hoc repair and distributional fit. |

For conditional work, also include kernel/nearest-neighbor conditional
resampling and a generator sharing our encoder/decoder. Use common paired
splits, held-out covariates, training/tuning budgets, and sampling budgets.
Report conditional object-space distances, event probabilities/calibration,
diversity, validity, and cost. Pooled W2 and the predicted Frechet mean alone
cannot establish conditional-law recovery.

### Citation register for all 42 works in the supplied report

This preserves the report's full reading scope. **F** means relevant
full-text sections were checked in this or the preceding review; **A** means
primary abstract/metadata was checked, without a full algorithm/proof audit;
**R** means retained from the supplied report as a reading lead, not
independently verified here. Neither F nor A means a reproduction.
The report provides full bibliographic entries; links below identify the
primary sources. The active local manuscript now cites 31 selected works,
including three relevant additions outside this report (SO(3) conditional
diffusion, FunDPS, and Global Frechet Manifold Learning).

| Work | Evidence | Role |
|---|---|---|
| [Meta Flow Matching](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ebdb990471f653dffb425eff03c7c980-Abstract-Conference.html) | A | Population dynamics conditioned on an initial-population representation. |
| [Deep Generative Models for Phylogenetic Inference](https://arxiv.org/abs/2610.19053) | A | Conditional discrete diffusion for whole trees, not fixed-star points. |
| [Latent Process Generator Matching](https://arxiv.org/abs/2605.20547) | A | Projected latent Markov processes; external-covariate interface not established by this check. |
| [Conditional Wasserstein Distances](https://jmlr.org/papers/v26/24-0586.html) | A | Conditional risk versus joint-law transport. |
| [Flow Matching on General Geometries](https://arxiv.org/abs/2302.03660) | A | Foundational geometric flow comparator. |
| [DiffeoCFM](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5616112a0120c15bf7d47a6bccc21bc3-Abstract-Conference.html) | A | Matrix pullback geometry and transformed-coordinate generation. |
| [DiffDock](https://arxiv.org/abs/2210.01776) | A | Conditional docking poses. |
| [Riemannian Score-Based Generative Modelling](https://arxiv.org/abs/2202.02763) | A | Intrinsic sphere diffusion candidate. |
| [Fast and Robust Visuomotor Riemannian Flow Matching Policy](https://arxiv.org/abs/2412.10855) | A | Visuomotor-conditioned manifold action generation; not a fixed-star baseline. |
| [PhylaFlow](https://arxiv.org/abs/2605.21859) | A | Whole-tree BHV transport; posterior recovery remains qualified in its abstract. |
| [Distribution-Conditioned Transport](https://arxiv.org/abs/2603.04736) | A | Transport conditioned on source/target distribution representations; not automatically scalar-covariate random-measure regression. |
| [Wasserstein Flow Matching](https://proceedings.mlr.press/v267/haviv25a.html) | F | Externally conditioned measure-valued generation. |
| [RWEFM](https://arxiv.org/abs/2609.25659) | A | Measures on manifolds; external interfaces not independently audited. |
| [Generator Matching](https://arxiv.org/abs/2410.20587) | A | General Markov-process construction; endpoint-conditioned generators do not establish external-covariate regression. |
| [Equivariant Diffusion for Molecule Generation in 3D](https://proceedings.mlr.press/v162/hoogeboom22a.html) | A | Structured atomic coordinates/types; application-level precedent. |
| [SE(3) pose diffusion](https://arxiv.org/abs/2305.15873) | A | Image-conditioned ambiguous object poses. |
| [Riemannian Diffusion Models](https://arxiv.org/abs/2208.07949) | A | Intrinsic manifold diffusion candidate. |
| [Sequence-Augmented SE(3)-Flow Matching](https://proceedings.neurips.cc/paper_files/paper/2024/hash/39ca8893ea38905a9d2ffe786e85af0f-Abstract-Conference.html) | A | Sequence-conditioned protein backbone generation. |
| [Bures-Wasserstein Flow Matching for Graph Generation](https://arxiv.org/abs/2506.14020) | A | Geometry-aware generation of entire graphs, not points on a fixed graph. |
| [Torsional Diffusion](https://arxiv.org/abs/2206.01729) | A | Molecular conformers via hypertorus diffusion. |
| [Dynamic Conditional Optimal Transport through Simulation-Free Flows](https://proceedings.neurips.cc/paper_files/paper/2024/hash/aa1d8cf866c4a684ef2e066dd200f8e4-Abstract-Conference.html) | A | Conditional transport, including infinite-dimensional settings. |
| [Functional Flow Matching](https://proceedings.mlr.press/v238/kerrigan24a.html) | A | Function-space flow construction; functions are not automatically probability-measure-valued outputs. |
| [Financial correlation-matrix DDPM](https://openaccess.city.ac.uk/id/eprint/34037/) | A | Explicit unconditional and conditional matrix generators. |
| [Infinite Brush](https://arxiv.org/abs/2407.14709) | A | External conditioning via a function-space cross-attention operator; ECCV 2024. |
| [SPD-DDPM](https://arxiv.org/abs/2312.08200) | F | Conditional SPD diffusion and generated-sample Frechet mean. |
| [Unlocking Point Processes through Point Set Diffusion](https://arxiv.org/abs/2410.22493) | F | General-metric point configurations; mask-conditioned completion/forecasting; Euclidean location-model implementation inspected in the paper. |
| [Edit-Based Flow Matching for Temporal Point Processes](https://arxiv.org/abs/2510.06050) | A | Conditional event-sequence generation reported; exact external-covariate interface not inspected. |
| [CorrFlow](https://arxiv.org/abs/2610.13606) | F | Continuous/categorical conditioning in correlation coordinates. |
| [Geometric Neural Diffusion Processes](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a797c2d2e0c1fdabf4d1ab8cd0b465c6-Abstract.html) | A | Conditional fields with Euclidean and spherical codomains. |
| [Generalized Wasserstein Flow Matching](https://arxiv.org/abs/2605.08424v3) | A | Probability measures over probability measures; nested couplings. External-covariate interface not established here. |
| [DeFoG](https://proceedings.mlr.press/v267/qin25d.html) | A | Conditional generation of entire graphs; not a fixed-star-point comparator. |
| [Minimax Optimal Rates for Regression on Manifolds and Distributions](https://arxiv.org/abs/2506.07504) | F | Conditional-law estimation with Holder-IPM loss and smooth embedded-manifold assumptions; not a general metric W2 theorem. |
| [Tang-Lin-Yang conditional diffusion](https://arxiv.org/abs/2409.20124) | F | Manifold-adaptive conditional estimation theory. |
| [DiGress](https://arxiv.org/abs/2209.14734) | A | Discrete graph diffusion with graph-level conditioning. |
| [RFdiffusion](https://pmc.ncbi.nlm.nih.gov/articles/PMC10468394/) | F | Conditional protein structure generation. |
| [Twisted Diffusion Sampler](https://arxiv.org/abs/2306.17775) | F | Conditional sampling from an existing diffusion model. |
| [Polynomial Convergence of Riemannian Diffusion Models](https://arxiv.org/abs/2601.02499) | A | Score-error/geometric convergence assumptions; exact smoothed target requires theorem inspection. |
| [PerturbDiff](https://arxiv.org/abs/2602.19685) | F | Conditional laws of distribution-valued objects in an embedding. |
| [MatterGen](https://www.nature.com/articles/s41586-025-08628-5) | A | Property-conditioned crystal generation; author preprint and Crossref publication metadata checked. |
| [Michelangelo](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ea1a7f7bc0fc14142106a84c94c826d0-Abstract-Conference.html) | A | Image/text-conditioned latent diffusion and a shape decoder; primary proceedings checked. |
| [Flow-based Generative Models for Learning Manifold to Manifold Mappings (DUALFLOW)](https://arxiv.org/abs/2012.10013) | A | Two-stream invertible GLOW for conditional manifold-valued modality transfer; AAAI 2021, not flow matching. |
| [Riemannian MeanFlow](https://arxiv.org/abs/2603.10718) | A | One-step manifold generation with classifier-free guidance; exact covariate type not checked. |

No R entries remain, but the A entries still require relevant full-text and
implementation checks before a narrow novelty claim or reproduction.
Citations were integrated into the Introduction,
related-work appendix, and active `thebibliography` in local
`overleaf/main.tex`. Updating `references.bib` alone would not affect this
manuscript's current citation renderer. These are local source edits, not a
live Overleaf upload.

### Final source-checking pass: scope and qualifications

The fourteen former reading leads were checked on 2026-11-01. Their primary
links are in the register above. The following notes record what the check
does and does not support; a missing conditioning claim in an abstract is
not evidence that the full method cannot condition.

| Work | Inspected evidence and qualification |
|---|---|
| Meta Flow Matching | ICLR abstract: amortized population dynamics using the initial population's representation. Distribution-valued inputs do not by themselves establish a random distribution-valued response at a scalar condition. |
| Latent Process Generator Matching | arXiv abstract: a projected tractable latent Markov process. No external-covariate implementation was verified. |
| Fast and Robust Visuomotor Riemannian Flow Matching Policy | arXiv abstract: externally conditioned visuomotor actions on Riemannian spaces. Relevant conditioning precedent, not an implementation for a branching metric star. |
| Distribution-Conditioned Transport | arXiv abstract: transport using distribution-level conditioning. Access to a target-distribution embedding is a different information regime from predicting an unknown random measure using only X. |
| Generator Matching | Abstract and introductory framing: target/endpoint-conditioned generators underlie a marginal generator. Conservatively classified A; no external-covariate algorithm was inspected. |
| Sequence-Augmented SE(3)-Flow Matching | NeurIPS abstract: sequence-conditioned protein backbone generation. Do not equate its structured output with a single point on a fixed network. |
| Bures-Wasserstein Flow Matching for Graph Generation | The supplied ICLR page was inaccessible; the author preprint, arXiv:2506.14020, establishes whole-graph generation via MRF representations. Specific conditioning interfaces remain unverified here. |
| Functional Flow Matching | PMLR abstract: flows over function space. A generated function need not be a normalized probability measure; this is not automatically our outer-law target. |
| Edit-Based Flow Matching for Temporal Point Processes | arXiv abstract reports conditional and unconditional event-sequence tasks. The condition's precise type was not checked in the algorithms. |
| Generalized Wasserstein Flow Matching | arXiv v3 abstract explicitly concerns probability measures over probability measures and nested couplings. This matches the object level of our measure study, not yet its scalar-X conditioning interface. |
| DeFoG | PMLR abstract reports conditional and unconditional graph generation. This is a citation for whole graphs, not a numerical baseline for points on our fixed star. |
| Minimax Optimal Rates for Regression on Manifolds and Distributions | Full-text Introduction and Section 2.2: the target is the conditional law, with Holder-IPM loss, smooth embedded supports, and separate covariate/response regularity. The paper describes TV and W1 cases; this is not our general-metric W2 decoder bound. No theorem/proof audit or automatic extension to random measures is claimed. |
| DUALFLOW | The AAAI endpoint was inaccessible, but the [author preprint](https://arxiv.org/abs/2012.10013) confirms two-stream GLOW modality transfer and AAAI 2021 acceptance. The citation is valid; its date is not a mismatch because normalizing flows predate flow matching. |
| Riemannian MeanFlow | arXiv abstract reports classifier-free guidance and one-step manifold generation. Neither a specific continuous-covariate interface nor support for nonsmooth branching geometry was verified. |

An arXiv submission contact is not an author list. Where page extraction
omitted authors, this pass did not replace the report's author metadata
with the submitter name. The earlier bibliography remains unchanged.

**Final baseline recommendation:** start with a known-law conditional
random-measure simulation and a complementary fixed-star-point simulation.
Use conditional resampling, direct conditional DDPM, and a same-encoder,
same-decoder conditional flow control. For the measure study, prioritize
the WFM family for a matched geometric comparison; PerturbDiff requires
a separately matched population/control observation model. For SPD data,
SPD-DDPM and DiffeoCFM remain a later alternative, not an additional
mandatory application. Whole-graph generators and smooth-manifold methods
are not valid plug-in star-point baselines. These recommendations specify
experiments to design, not results or proven literature gaps.

## Suggested collaborator update

> I completed a primary-source pass over all 42 references, with nine
> relevant full-text checks and 33 abstract/metadata checks, not reproductions
> or proof audits. SPD-DDPM and PerturbDiff are particularly close; PerturbDiff already
> generates conditional distribution-valued outputs through an embedding.
> WFM also has externally conditioned distribution-valued experiments.
> I have added the central references to the local manuscript and retained the
> report's full 42-work reading list with evidence levels. For our current
> experiments, the main missing specialized comparisons are intrinsic
> Riemannian diffusion on the sphere and WFM for distribution-valued objects.
> SPD-DDPM/DiffeoCFM become relevant if we add an SPD benchmark. Whole-graph
> and phylogenetic-tree generators are not direct baselines for points on our
> fixed star. The contribution should be investigated in the precise metric
> formulation and recovery analysis, not conditioning alone.
> I also prepared a conditional experimental protocol with held-out
> covariates, conditional resampling, matched generator controls, object-space
> distributional evaluation, and event calibration. No conditional training
> or external baseline reproduction has started.

Saved experimental evidence is reported separately in
[Experimental Results and Completion Status](experimental_results_and_completion_status.md). No training
was restarted for this literature review or results inspection.
