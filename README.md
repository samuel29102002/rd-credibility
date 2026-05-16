# rd-credibility

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

**Credibility diagnostics and replication audits for regression discontinuity designs.**

Point-and-click dashboard + Python library that takes any RD dataset and answers one question: *should we believe this estimate?*

---

## Why this exists

The replication crisis has hit quasi-experimental methods hard. RD designs in particular are vulnerable to:
- Manipulation of the running variable near the threshold
- Researcher degrees of freedom in bandwidth selection
- Fragile estimates that disappear at slightly different bandwidths
- Overlooked covariate imbalance and placebo failures

Most practitioners run a McCrary test and call it done. `rd-credibility` runs *all four* diagnostics, synthesises them into a single grade, and — when auditing a published paper — tells you exactly which dimensions failed and why.

---

## Features

- **Estimate** — local linear/polynomial RD with HC1 standard errors and CCT MSE-optimal bandwidth
- **Density test** — McCrary (2008) manipulation test with binwidth-robust implementation
- **Covariate balance** — per-covariate RD tests with family-wise multiple testing correction
- **Placebo cutoffs** — grid of false cutoffs to check specificity of the effect
- **Bandwidth sensitivity** — coefficient-of-variation score across a bandwidth grid
- **Donut robustness** — excludes close-to-cutoff observations to detect local manipulation
- **Credibility score** — composite 0–100 index with letter grade (A–F)
- **Replication audit** — upload a published specification; receive a Robust / Fragile / Problematic verdict
- **Dashboard** — 7-page Streamlit app with interactive Plotly + publication-ready Matplotlib figures

---

## Quickstart

```bash
pip install rd-credibility
streamlit run rd_credibility/app/main.py
```

Or use the library directly:

```python
from rd_credibility.app.components.data_loader import load_builtin
from rd_credibility.estimation.bandwidth import mse_optimal_bandwidth
from rd_credibility.estimation.rdrobust import RDEstimator
from rd_credibility.diagnostics.mccrary import McCraryTest
from rd_credibility.scoring.credibility import CredibilityScore

df = load_builtin("Lee 2008 (Electoral)")
y, x = df["y"].values, df["x"].values
bw = mse_optimal_bandwidth(y, x, cutoff=0.0)
rd = RDEstimator(y, x, cutoff=0.0, bandwidth=bw).fit()
mccrary = McCraryTest(x, cutoff=0.0).fit()
print(f"Estimate: {rd.estimate:.3f}  p(manipulation)={mccrary.p_value:.3f}")
```

Full walkthrough: [docs/quickstart.md](docs/quickstart.md)

---

## Credibility score

| Component | Max points | Failure conditions |
|-----------|-----------|-------------------|
| McCrary density test | 30 | Warn: −10; Fail: −30; p<0.01 caps score at 25 |
| Covariate balance | 25 | 1 imbalanced: −10; ≥2: −25 |
| Bandwidth sensitivity | 25 | CV ≥ 0.10: −5 to −25 |
| Placebo cutoffs | 20 | 1–2 significant: −8; ≥3: −20 |

Grades: A (90–100) · B (75–89) · C (60–74) · D (45–59) · F (0–44)

---

## Built-in datasets

| Dataset | N | True τ | Source |
|---------|---|--------|--------|
| Lee 2008 (Electoral) | ~6,558 | ~0.40 | Lee (2008) |
| Maimonides Rule | ~2,019 | varies | Angrist & Lavy (1999) |
| Thistlethwaite 1960 | 3,000 | ~0.50 | Thistlethwaite & Campbell (1960) |

All datasets are simulated analogues with the same identifying variation as the originals.

---

## Replication audit

```python
from rd_credibility.app.components.replication import run_replication_audit

audit = run_replication_audit(
    y, x, cov_cols=[], cov_data=None,
    cutoff=0.0,
    reported_bandwidth=0.10,
    reported_estimate=0.08,
    reported_se=0.02,
)
print(audit.verdict)         # "Robust" | "Fragile" | "Problematic"
print(audit.verdict_reasons) # list of plain-English failure strings
```

The dashboard's Replication page (page 7) provides the same audit with a full UI including sensitivity heatmaps and downloadable HTML reports.

---

## Statistical methodology

See [docs/methodology.md](docs/methodology.md) for:
- RD identification assumption
- Local polynomial estimation and HC1 SE derivation
- CCT MSE-optimal bandwidth formula
- Each diagnostic test (statistic, null hypothesis, thresholds)
- Credibility score weighting rationale
- Full references

---

## References

- McCrary, J. (2008). Manipulation of the running variable in the regression discontinuity design. *Journal of Econometrics*, 142(2), 698–714.
- Calonico, S., Cattaneo, M.D., and Titiunik, R. (2014). Robust nonparametric confidence intervals for regression-discontinuity designs. *Econometrica*, 82(6), 2295–2326.
- Lee, D.S. (2008). Randomized experiments from non-random selection in U.S. House elections. *Journal of Econometrics*, 142(2), 675–697.
- Angrist, J.D. and Lavy, V. (1999). Using Maimonides' rule to estimate the effect of class size. *Quarterly Journal of Economics*, 114(2), 533–575.
- Thistlethwaite, D.L. and Campbell, D.T. (1960). Regression-discontinuity analysis. *Journal of Educational Psychology*, 51(6), 309–317.

---

## License

MIT © RD Credibility Team
