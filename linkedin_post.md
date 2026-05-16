Most published regression discontinuity papers report a point estimate, a McCrary test, and stop there — leaving readers no way to judge whether the result survives basic scrutiny.

I built **rd-credibility**: an open-source Python library and Streamlit dashboard that runs a full diagnostic battery on any RD design and collapses the evidence into a single 0–100 credibility score.

The four pillars: density manipulation test, pre-determined covariate balance, placebo cutoff specificity, and bandwidth sensitivity. The score reflects how much of the diagnostic literature — Cattaneo, Imbens, and more recently Roth on pre-trends — actually demands from a credible quasi-experiment.

The replication audit mode is the part I find most useful: paste in a published specification (estimate, SE, bandwidth), and the tool tells you whether it can reproduce the result, whether the bandwidth is suspiciously far from MSE-optimal, and whether the estimate holds up across nearby specifications.

Verdict comes back as Robust, Fragile, or Problematic — with plain-English reasons attached.

GitHub: https://github.com/your-org/rd-credibility — stars help with visibility.

#CausalInference #EconML #OpenSource
