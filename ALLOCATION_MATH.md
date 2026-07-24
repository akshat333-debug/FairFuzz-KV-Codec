# Minimax Allocation: Derivation and KKT Conditions

This document derives the continuous fairness-constrained allocation that
`fairfuzzkv_codec.allocation.minimax` implements. The implementation is checked
against this derivation (see `tests/allocation/test_minimax.py`).

## Problem

Cohorts `l = 1..L`. Each has an exponential distortion-vs-bits curve

    D_l(x_l) = alpha_l * exp(-beta_l * x_l),    alpha_l > 0, beta_l > 0

where `x_l` is the **total serialized bits** spent on cohort `l` (so `x_l`
already folds in the cohort's element count and any fixed codebook overhead is
part of the measured cost). Fixed total budget `B`.

**Frozen default objective — minimize the worst-cohort distortion:**

    minimize_x    max_l  D_l(x_l)
    subject to    sum_l x_l <= B,   x_l >= 0.

## Epigraph form

Introduce `t` = worst-case distortion:

    minimize_{x,t}   t
    subject to       D_l(x_l) <= t        for all l
                     sum_l x_l <= B
                     x_l >= 0.

## KKT conditions

Lagrangian (multipliers `mu_l >= 0` for the epigraph constraints, `lambda >= 0`
for the budget):

    Λ = t + sum_l mu_l (D_l(x_l) - t) + lambda (sum_l x_l - B)   (x_l>=0 handled by active set)

Stationarity:

  * ∂Λ/∂t   = 1 - sum_l mu_l = 0                    ⇒  **sum_l mu_l = 1**
  * ∂Λ/∂x_l = mu_l D_l'(x_l) + lambda = 0.
    With D_l'(x_l) = -beta_l * D_l(x_l):
                    lambda = mu_l * beta_l * D_l(x_l).

Complementary slackness:

  * budget tight when lambda > 0:  sum_l x_l = B.
  * for an **active** cohort (x_l > 0, mu_l > 0):  D_l(x_l) = t.
  * for an **inactive** cohort (x_l = 0):  mu_l may be 0 and D_l(0) = alpha_l <= t.

## What is actually equalized at the optimum

**The per-cohort DISTORTION of every active cohort is equalized to the common
worst-case value `t`:**

    D_l(x_l) = t     for all active l.

This is the precise statement. It is **NOT** true that "beta is equalized" — the
`beta_l` are fixed properties of each cohort's curve, not decision variables, and
nothing forces them equal. What the optimum equalizes is the **achieved
distortion** `D_l`, i.e. every active cohort is pushed down to the same worst-case
floor `t`; a cohort is left inactive precisely when it is already below that floor
at zero bits (`alpha_l <= t`).

The multipliers do carry a beta-weighting — `mu_l = lambda n_l /(beta_l t)`-style
— but that is the *sensitivity*, not the equalized quantity. Saying "the optimum
equalizes distortion across active cohorts" is correct; saying "it equalizes
beta" is not.

## Closed form on the active set

With `D_l(x_l) = t` on the active set `A`, take logs:

    alpha_l e^{-beta_l x_l} = t   ⇒   x_l = (ln alpha_l - ln t) / beta_l.

Impose the budget `sum_{l in A} x_l = B`, let `Λ = ln t`:

    sum_{l in A} (ln alpha_l - Λ)/beta_l = B
    ⇒  Λ = ( sum_{l in A} (ln alpha_l)/beta_l  -  B ) / ( sum_{l in A} 1/beta_l )
    ⇒  t = exp(Λ),   x_l = (ln alpha_l - Λ)/beta_l.

## Active-set iteration (water-filling)

Some `x_l` may come out **negative** (a cohort whose `alpha_l <= t`: it is already
better than the worst-case floor and should get 0 bits). Standard fix:

1. Solve the closed form over the current active set `A`.
2. If every `x_l >= 0`, done.
3. Otherwise drop the cohort with the most-negative `x_l` (set `x_l = 0`,
   distortion `alpha_l`), and re-solve on the smaller active set.
4. Repeat until all active `x_l >= 0`.

Because dropping a cohort only removes budget demand, the loop terminates in at
most `L` steps.

## Feasibility & monotonicity

* **Budget feasibility:** any `B >= 0` is feasible in the continuous problem
  (there is no upper cap on `x_l`; more bits only lower distortion). Infeasibility
  appears only after **discrete projection** or under hard **minimum-quality**
  constraints `D_l(x_l) <= q_l`, which can be unmeetable by the available options.
* **Monotonicity in budget:** the optimal worst-case `t*(B)` is non-increasing in
  `B` — more total bits can never make the best achievable worst-cohort distortion
  worse. Proven for the continuous solution (`Λ` is affine-decreasing in `B`, so
  `t = e^Λ` decreases) and tested for the discrete solver.

## Variants (item 75)

* **max distortion** (default, above).
* **weighted max:** replace `D_l` by `w_l D_l`; the same derivation gives
  `w_l D_l(x_l) = t` on the active set.
* **minimum-quality:** add hard constraints `D_l(x_l) <= q_l`; options violating
  `q_l` are removed before optimizing (may render the instance infeasible, which
  is reported, not fudged).

## Discrete projection (item 73)

The continuous `x_l*` are projected onto the actual discrete option menu
(scalar bit-widths / LBG configs) by selecting, per cohort, the cheapest option
whose distortion does not exceed the target floor `t`, then repairing any budget
overflow by downgrading the cohort with the smallest resulting distortion
increase. The exact discrete reference optimizer (`solve_minimax_exact`, a binary
search over candidate worst-case values) validates this projection on small
instances.
