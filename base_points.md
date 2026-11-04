# Base Points and Landmarks Used in the Initial Experiment

## Scope

These are the reference points used in the initial **two-moons and
eight-Gaussians pilot**, with random seed **0** and **4,096 training observations
per dataset**. This pilot is distinct from the subsequent five-seed experiments.

Each coordinate encoder used **one base point** `b` and **three landmark points**
`a1`, `a2`, and `a3`. All four were distinct observations from the training set.
They were fixed for the corresponding dataset and landmark configuration, not
selected separately for each generated sample.

## Base-Point Selection

For each dataset, we computed the coordinate-wise median of the training data
and selected the training observation nearest to that median in Euclidean
distance:

$$
b = \underset{x \in \mathcal{D}_{\mathrm{train}}}{\arg\min}
\left\|x-\operatorname{median}_{\mathrm{coordinate}}
(\mathcal{D}_{\mathrm{train}})\right\|_2^2.
$$

Thus, the base was an actual training observation, **not necessarily the origin
or the median itself**. The same base was used for all three landmark-selection
rules within a dataset and seed.

| Dataset | Base point `b` | Training-array index (zero-based) |
|---|---|---|
| Two moons | (0.65947, 0.52184) | 877 |
| Eight Gaussians | (-0.16308, 1.42432) | 3727 |

Coordinates in this document are rounded to five decimal places.

## Landmark-Selection Rules

The pilot compared three rules, each selecting three distinct training
observations and excluding the base:

1. **Manual directional extrema:** Select the observation with the largest
   projection of its displacement from the base along each of the predeclared
   directions **0, 120, and 240 degrees**, excluding previously selected points.
   The directions were fixed before examining results.
2. **Random:** Sample three training observations without replacement, excluding
   the base, using a seeded random generator.
3. **Farthest-point sampling (FPS):** Start with the base and repeatedly select
   the training observation whose distance to its nearest already-selected
   reference point is largest, until three landmarks have been added.

## Actual Landmarks for the Manual Configuration

The coordinates below apply to the **manual** rule only; random and FPS use
different landmarks.

| Dataset | Landmark | Direction | Coordinates | Training-array index (zero-based) |
|---|---|---|---|---|
| Two moons | `a1` | 0 degrees | (2.16701, 0.51741) | 2355 |
| Two moons | `a2` | 120 degrees | (-0.52732, 1.04811) | 1023 |
| Two moons | `a3` | 240 degrees | (-1.07919, -0.06222) | 3367 |
| Eight Gaussians | `a1` | 0 degrees | (2.50256, -0.14519) | 3383 |
| Eight Gaussians | `a2` | 120 degrees | (-1.80600, 1.77449) | 2259 |
| Eight Gaussians | `a3` | 240 degrees | (-1.74193, -1.72702) | 3009 |

## How the Reference Points Define the Coordinates

For an observation `x`, the three raw law-of-cosines coordinates are:

$$
z_j(x)=
\frac{d(x,b)^2+d(b,a_j)^2-d(x,a_j)^2}
{2d(b,a_j)}, \qquad j=1,2,3.
$$

In these Euclidean experiments, this is equivalent to projecting `x - b` onto
the unit direction from the base to each landmark:

$$
z_j(x)=(x-b)^\top\frac{a_j-b}{\|a_j-b\|_2}.
$$

The resulting three coordinates were standardized using their training-set
means and sample standard deviations before diffusion training. Both manual
configurations had landmark directions of rank 2, spanning the original
two-dimensional object space.

## Reproducibility Sources

- [Reference selection and encoder implementation](experiments/run_experiments.py)
- [Initial pilot configuration](experiments/results/pilot/config.json)
- [Two-moons manual reference points and encoder statistics](experiments/results/pilot/two_moons/seed_0/manual/references.json)
- [Eight-Gaussians manual reference points and encoder statistics](experiments/results/pilot/eight_gaussians/seed_0/manual/references.json)
