# Quickstart

## Installation

```bash
pip install rd-credibility          # from PyPI (once published)
# or from source:
git clone https://github.com/your-org/rd-credibility.git
cd rd-credibility
pip install -e .
```

Python 3.10+ is required.

---

## Running a full diagnostic suite in 5 lines

```python
import pandas as pd
from rd_credibility.app.components.data_loader import load_builtin
from rd_credibility.estimation.bandwidth import mse_optimal_bandwidth
from rd_credibility.estimation.rdrobust import RDEstimator
from rd_credibility.diagnostics.mccrary import McCraryTest
from rd_credibility.diagnostics.balance import CovariateBalance
from rd_credibility.diagnostics.placebo import PlaceboTest
from rd_credibility.diagnostics.sensitivity import BandwidthSensitivity
from rd_credibility.scoring.credibility import CredibilityScore

# 1. Load data
df = load_builtin("Lee 2008 (Electoral)")
y, x = df["y"].values, df["x"].values
cutoff = 0.0

# 2. Select bandwidth and estimate
bw = mse_optimal_bandwidth(y, x, cutoff)
rd = RDEstimator(y, x, cutoff=cutoff, bandwidth=bw).fit()
print(f"Estimate: {rd.estimate:.3f}  SE: {rd.se:.3f}  95% CI: [{rd.ci_lower:.3f}, {rd.ci_upper:.3f}]")

# 3. Run diagnostics
mccrary  = McCraryTest(x, cutoff=cutoff).fit()
covs     = df[[c for c in df.columns if c.startswith("z")]].values
balance  = CovariateBalance(x, covs, cutoff=cutoff, bandwidth=bw).fit()
placebo  = PlaceboTest(y, x, cutoff=cutoff).fit()
bw_grid  = BandwidthSensitivity(y, x, cutoff=cutoff).fit()

# 4. Compute credibility score
report = CredibilityScore(
    mccrary_result=mccrary,
    balance_result=balance,
    bandwidth_result=bw_grid,
    placebo_result=placebo,
).compute()

print(f"Score: {report.score}/100  Grade: {report.grade}")
print(f"McCrary p-value: {mccrary.p_value:.3f}")
```

---

## Loading built-in datasets

Three empirical/simulated benchmark datasets are shipped with the package:

```python
from rd_credibility.app.components.data_loader import load_builtin

df = load_builtin("Lee 2008 (Electoral)")   # n≈6558, y∈[0,1], tau≈0.40
df = load_builtin("Maimonides Rule")        # n≈2019, class-size discontinuity
df = load_builtin("Thistlethwaite 1960")    # n=3000, scholarship threshold
```

Each DataFrame has columns `x` (running variable), `y` (outcome), `d` (treatment indicator), and zero or more `z_*` (covariates).

---

## Using your own data

```python
import pandas as pd

df = pd.read_csv("my_data.csv")

# Rename columns to match the expected format
y = df["outcome"].values
x = df["score"].values   # running variable, centred so cutoff = 0
cutoff = 0.0             # or whatever your threshold is

from rd_credibility.estimation.bandwidth import mse_optimal_bandwidth
bw = mse_optimal_bandwidth(y, x, cutoff)
```

---

## Visualising the discontinuity

### Interactive (Plotly)

```python
from rd_credibility.visualization.rd_plot import plot_interactive

fig = plot_interactive(y, x, cutoff=cutoff, bandwidth=bw, n_bins=40)
fig.show()                          # opens in browser
fig.write_html("rd_plot.html")      # save as standalone HTML
```

### Publication-ready (Matplotlib)

```python
from rd_credibility.visualization.rd_plot import plot_publication

fig = plot_publication(y, x, cutoff=cutoff, bandwidth=bw, n_bins=30)
fig.savefig("figure1.pdf", bbox_inches="tight", dpi=300)
```

### Diagnostic panel

```python
from rd_credibility.visualization.diagnostic_plots import plot_diagnostic_panel

fig = plot_diagnostic_panel(
    mccrary_result=mccrary,
    balance_result=balance,
    bw_result=bw_grid,
    placebo_result=placebo,
)
fig.savefig("diagnostics.pdf", bbox_inches="tight")
```

---

## Launching the dashboard

```bash
streamlit run rd_credibility/app/main.py
```

The dashboard opens at `http://localhost:8501` and provides:
- Interactive RD plot with adjustable bandwidth
- Four diagnostic tabs (density, balance, sensitivity, placebo)
- Credibility score gauge
- Replication audit mode (page 7) for auditing published results

---

## Running a replication audit

Programmatically audit a published RD specification:

```python
from rd_credibility.app.components.replication import run_replication_audit

audit = run_replication_audit(
    y, x,
    cov_cols=["z_lag_vote", "z_turnout"],
    cov_data=df[["z_lag_vote", "z_turnout"]].values,
    cutoff=0.0,
    reported_bandwidth=0.10,    # as stated in the paper
    reported_estimate=0.08,     # point estimate in the paper
    reported_se=0.02,           # SE in the paper
)

print(f"Verdict: {audit.verdict}")           # Robust / Fragile / Problematic
print(f"Score:   {audit.credibility_score}")
print(f"Grade:   {audit.credibility_grade}")
print(f"Issues:  {audit.verdict_reasons}")
```

---

## Running tests

```bash
# Fast tests only (< 15 seconds)
python -m pytest tests/ -m "not slow" -q

# Full suite including Monte Carlo (several minutes)
python -m pytest tests/ -q

# Coverage report
pip install pytest-cov
python -m pytest tests/ -m "not slow" --cov=rd_credibility --cov-report=term-missing -q
```

---

## Next steps

- Read `docs/methodology.md` for the statistical background behind each diagnostic.
- See the inline docstrings (`help(RDEstimator)`, `help(CredibilityScore)`) for full API documentation.
- File issues or contribute at the project repository.
