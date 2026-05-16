# Methodology

## 1. Regression Discontinuity Design

A sharp regression discontinuity (RD) design exploits a threshold rule that assigns treatment:

```
D_i = 1(X_i ≥ c)
```

where `X_i` is the running variable, `c` is the known cutoff, and `D_i ∈ {0,1}` is binary treatment.

The parameter of interest is the **average treatment effect at the cutoff**:

```
τ = lim_{x↓c} E[Y | X=x] − lim_{x↑c} E[Y | X=x]
```

### Identification Assumption

The potential outcomes `(Y(0), Y(1))` are **continuously distributed** in the running variable at the cutoff. Equivalently, the conditional regression functions `E[Y(0)|X=x]` and `E[Y(1)|X=x]` are continuous at `c`. This implies that any discontinuity in the observed outcome `E[Y|X=x]` at `c` is attributable solely to the treatment.

This assumption fails if agents **manipulate** the running variable to sort across the threshold, because then units just above and below `c` are systematically different in unobservables.

---

## 2. Estimation: Local Polynomial Regression

### Estimator

`rd-credibility` uses **local polynomial regression** of order `p` (default `p=1`, i.e., local linear). Separate polynomials are fit on each side of the cutoff using **kernel-weighted** observations within a bandwidth `h`:

```
τ̂ = μ̂₊(c) − μ̂₋(c)
```

where `μ̂₊(c)` and `μ̂₋(c)` are the right- and left-side fitted values at the cutoff.

The kernel weights default to the **triangular kernel** `K(u) = (1−|u|)·1(|u|≤1)`, which is MSE-optimal for local linear regression. Uniform and Epanechnikov kernels are also supported.

### Standard Errors

Heteroskedasticity-consistent (HC1) sandwich standard errors are computed:

```
Var(τ̂) = (X'WX)⁻¹ (X'W diag(ê²·n/(n−p−1)) WX) (X'WX)⁻¹
```

evaluated at both sides and combined. This is robust to arbitrary conditional heteroskedasticity near the cutoff.

---

## 3. Bandwidth Selection: CCT MSE-Optimal

The bandwidth `h` controls the bias-variance trade-off. The default bandwidth minimizes the **mean squared error** of the local linear estimator, following Calonico, Cattaneo, and Titiunik (2014).

The CCT plug-in procedure:
1. Estimate a preliminary bandwidth using rule-of-thumb regularisation
2. Estimate bias (using local quadratic) and conditional variance at the cutoff on each side
3. Compute the MSE-optimal bandwidth:

```
h* = C(K) · [ σ²₊/n·f(c) + σ²₋/n·f(c) ]^(1/5) / B²^(1/5)
```

where `C(K)` is a kernel constant, `σ²±` are conditional variances at the cutoff, `f(c)` is the running-variable density, and `B` is the leading bias term.

**Why this matters for credibility:** A reported bandwidth that deviates substantially from `h*` is a credibility concern. Using `h >> h*` over-smooths (increases bias); `h << h*` under-smooths (inflates standard errors and may produce spurious precision).

---

## 4. Diagnostic Tests

### 4.1 McCrary Density Test

**Purpose:** Test whether units manipulate the running variable to cross the threshold.

**Statistic:** McCrary (2008) fits separate local linear regressions to the empirical density histogram on each side of the cutoff and tests for a discontinuity in the log-density:

```
H₀: f(c⁺) = f(c⁻)   (density is continuous at the cutoff)
```

The test statistic is a t-ratio of the estimated log-density gap divided by its standard error.

**Thresholds used by `rd-credibility`:**

| p-value | Status | Credibility impact |
|---------|--------|-------------------|
| > 0.15 | Pass | No deduction |
| 0.05 – 0.15 | Warn | Moderate deduction |
| < 0.05 | Fail | Hard cap: score ≤ 25 if p < 0.01 |

**Reference:** McCrary, J. (2008). Manipulation of the running variable in the regression discontinuity design: A density test. *Journal of Econometrics*, 142(2), 698–714.

---

### 4.2 Covariate Balance

**Purpose:** Test whether pre-determined covariates are balanced across the cutoff. Since covariates determined before treatment assignment cannot be affected by treatment, a discontinuity in covariates at `c` indicates that the RD assumption is violated.

**Procedure:** For each covariate `Z_k`, run the same local linear regression as for the outcome and test `H₀: E[Z_k | X=c⁺] = E[Z_k | X=c⁻]`.

**Thresholds:**

| Significant covariates | Status |
|------------------------|--------|
| 0 | Pass |
| 1 | Warn |
| ≥ 2 | Fail |

**Reference:** Lee, D.S. and Lemieux, T. (2010). Regression discontinuity designs in economics. *Journal of Economic Literature*, 48(2), 281–355.

---

### 4.3 Placebo Cutoff Test

**Purpose:** Test whether the estimated effect is specific to the true cutoff or appears throughout the running-variable distribution (which would suggest spurious effects).

**Procedure:** Estimate the RD effect at a grid of placebo cutoffs drawn from the interior of each side (never within one bandwidth of the true cutoff). Under `H₀: τ(c_placebo) = 0`, the fraction of significant placebo effects should be ≤ 5%.

**Thresholds:**

| Significant placebos | Status |
|---------------------|--------|
| 0 out of `n` | Pass |
| 1–2 | Warn |
| ≥ 3 | Fail |

---

### 4.4 Bandwidth Sensitivity

**Purpose:** Check whether the estimate is robust across a reasonable range of bandwidths. A credible RD effect should not depend strongly on the exact bandwidth chosen.

**Procedure:** Estimate the RD effect on a grid of bandwidths `h ∈ [0.5h*, 2.0h*]` (25 grid points by default). The **coefficient of variation** (CV) of estimates across this grid measures sensitivity:

```
CV = std(τ̂(h)) / |mean(τ̂(h))|
```

**Thresholds:**

| CV | Status |
|----|--------|
| < 0.10 | Stable |
| 0.10 – 0.20 | Moderate |
| 0.20 – 0.50 | Sensitive |
| ≥ 0.50 | Highly sensitive |

---

### 4.5 Donut Robustness

**Purpose:** Check whether observations very close to the cutoff are driving the result. Units near the cutoff are most susceptible to manipulation and measurement error. Removing a small "donut hole" around the cutoff and re-estimating provides a robustness check.

**Procedure:** Estimate the RD effect excluding observations within radius `r` of the cutoff, for `r ∈ {0, 0.02, 0.05}` (in running-variable units). Estimates should not change drastically.

---

## 5. Credibility Score

The credibility score aggregates all diagnostic evidence into a single 0–100 index. Each component contributes a maximum number of points:

| Component | Max points | Deduction triggers |
|-----------|-----------|-------------------|
| McCrary density test | 30 | Warn: −10; Fail: −30 |
| Covariate balance | 25 | 1 imbalanced: −10; ≥2: −25 |
| Bandwidth sensitivity | 25 | CV 0.10–0.20: −5; 0.20–0.50: −15; ≥0.50: −25 |
| Placebo cutoffs | 20 | 1–2 significant: −8; ≥3: −20 |

**Hard ceiling:** If the McCrary p-value < 0.01 (strong evidence of manipulation), the score is capped at 25 regardless of other diagnostics.

### Grade boundaries

| Score | Grade |
|-------|-------|
| 90–100 | A |
| 75–89 | B |
| 60–74 | C |
| 45–59 | D |
| 0–44 | F |

---

## 6. Replication Audit

When auditing a published RD specification, `rd-credibility` runs four additional checks:

### Estimate Reproduction

Compares the user-reported estimate to the re-estimated value. Discrepancy is measured in units of the reported standard error:

```
diff_in_SE = |τ̂_ours − τ̂_reported| / SE_reported
```

| diff_in_SE | Status |
|-----------|--------|
| < 1.0 | Reproduced |
| 1.0 – 2.0 | Marginal |
| ≥ 2.0 | Failed |

### Bandwidth Choice

The ratio `h_reported / h*_MSE` is examined:

| Ratio | Concern |
|-------|---------|
| 0.5 – 2.0 | Acceptable |
| > 3.0 | Over-smoothed (high bias risk) |
| < 0.30 | Under-smoothed (variance unstable) |

### Specification Fragility

Estimates are computed across a fine grid of bandwidths near the reported value. The fragility score is the fraction of nearby bandwidths giving estimates within 1.96 standard errors of the median:

```
fragility_score = (# stable BWs) / (# total BWs in ±50% of reported BW)
```

| Score | Status |
|-------|--------|
| ≥ 0.80 | Stable |
| 0.50 – 0.80 | Moderate |
| < 0.50 | Fragile |

### Overlooked Diagnostics

Checks whether key diagnostic tests — McCrary, covariate balance, placebo — were conducted and what they showed.

### Overall Verdict

| Verdict | Conditions |
|---------|-----------|
| **Robust** | All four dimensions pass |
| **Fragile** | Any dimension shows moderate concern |
| **Problematic** | Estimate not reproduced OR McCrary/balance fail |

---

## 7. Limitations

- The density continuity assumption (McCrary) is **necessary but not sufficient** for valid RD. It can miss sorting strategies that preserve density continuity.
- Covariate balance tests have **limited power** when covariate data is unavailable or when the effect is small relative to covariate variation.
- The credibility score weights are **calibrated judgments**, not derived from a formal decision-theoretic criterion. Different applied contexts may warrant different weighting.
- The CCT bandwidth selector assumes the local polynomial model is correctly specified near the cutoff. In the presence of a kink exactly at the cutoff, the bias estimate can be misleading.
- All results are **sample-size dependent**. A clean small dataset may score poorly simply because tests have low power.

---

## 8. References

- Angrist, J.D. and Lavy, V. (1999). Using Maimonides' rule to estimate the effect of class size on scholastic achievement. *Quarterly Journal of Economics*, 114(2), 533–575.
- Calonico, S., Cattaneo, M.D., and Titiunik, R. (2014). Robust nonparametric confidence intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295–2326.
- Imbens, G.W. and Lemieux, T. (2008). Regression discontinuity designs: A guide to practice. *Journal of Econometrics*, 142(2), 615–635.
- Lee, D.S. (2008). Randomized experiments from non-random selection in U.S. House elections. *Journal of Econometrics*, 142(2), 675–697.
- Lee, D.S. and Lemieux, T. (2010). Regression discontinuity designs in economics. *Journal of Economic Literature*, 48(2), 281–355.
- McCrary, J. (2008). Manipulation of the running variable in the regression discontinuity design: A density test. *Journal of Econometrics*, 142(2), 698–714.
- Thistlethwaite, D.L. and Campbell, D.T. (1960). Regression-discontinuity analysis: An alternative to the ex post facto experiment. *Journal of Educational Psychology*, 51(6), 309–317.
