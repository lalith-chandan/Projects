# IPL Powerplay Prediction

**Course**: EE3111 — Statistics for Electrical Engineers  
**Semester**: Jan–May 2026

## Problem

Predict the total runs scored by the batting team at the end of 6 overs (Powerplay) in an IPL T20 match, given only the ball-by-ball data from the first 3 overs and match metadata.

- **Input**: Two CSV strings (Ashwin/Cricsheet format) — match metadata + ball-by-ball data up to 3 overs
- **Output**: A single float — predicted runs at 6 overs
- **Metric**: RMSE (Root Mean Squared Error), evaluated on live IPL 2026 matches

## Final Model

Unweighted multivariate linear regression (OLS) with 5 features and one non-linear transformation:

```
Y = 1.817·x₁ + 1.100·x₂ - 1.858·x₃ + 1.063·e^(-x₄) + 1.292·x₅ + 6.380
```

| Feature | Description | Coefficient |
|---------|-------------|-------------|
| x₁ | Runs scored off the bat (excl extras) in first 3 overs | +1.817 |
| x₂ | Number of dot balls in first 3 overs | +1.100 |
| x₃ | Number of boundaries (4s + 6s) in first 3 overs | -1.858 |
| x₄ | Number of wickets in first 3 overs (transformed as e^(-x₄)) | +1.063 |
| x₅ | Total extras (wides, no-balls) in first 3 overs | +1.292 |

**RMSE**: 9.08 (full dataset), 9.20 ± 0.69 (cross-validated on 200 random 80-20 splits)

## Experiments Conducted

The `experiments.ipynb` notebook documents every experiment in detail:

1. **Data selection**: Why 2020+ (not 2008+) — validated that older data increases RMSE
2. **Scatter plot analysis**: Identified linear trends for runs/dots/boundaries, exponential decay for wickets
3. **Per-team vs pooled model**: Per-team models overfit (coefficient instability with ~35 matches each). Pooled model wins.
4. **Extras as separate feature**: Splitting batting runs and extras improved RMSE by ~0.9
5. **Wicket transformation**: Compared e^(-x), 2^(-x), and linear — all identical within 0.001. Kept e^(-x) for theoretical justification.
6. **Momentum indicator**: Linear slope of runs across overs 1-2-3. Tiny improvement (~0.04), rejected for complexity.
7. **Wicket timing**: Recency-weighted wicket score. Hypothesis confirmed in data but hurt regression RMSE.
8. **Venue/stadium weight**: Historical venue averages. Increased RMSE — training venue stats don't predict test conditions.
9. **Team-specific intercepts**: One-hot team biases. Hurt RMSE — rosters change between seasons.
10. **Pairwise combinations**: All combos of momentum + wicket timing + venue tested. None helped.
11. **Time-weighted OLS**: 6 different weighting schemes. Best (gentle) matched unweighted on test data. Rejected for simplicity.
12. **Log-linear model**: exp(c + w·x) — worse RMSE because prediction errors are additive, not multiplicative.
13. **Feature ablation**: All 31 feature combinations tested. runs_excl_extras is dominant; just runs+extras gets 9.21.
14. **Baseline comparisons**: 2× runs at 3 overs gives RMSE 12.00; bias-corrected 11.11. Our model: 9.08.

## Repository Structure

```
├── README.md                  # This file
├── main.py                    # Final submission file (predict function)
├── experiments.ipynb           # Complete experiment notebook with all code and reasoning
├── EE3111_Project_Report.pdf  # Formal PDF report (16 pages)
├── data/
│   └── ipl_features.csv       # Pre-extracted feature dataset (419 matches, 2020-2026)
└── plots/
    ├── scatter_plots.png       # Feature vs target scatter plots with linear fits
    └── box_plots.png           # Box plots showing outliers across feature bins
```

## How to Run

```python
from main import predict

# Pass match_data CSV string and ball_by_ball_data CSV string
predicted_runs = predict(match_csv_string, ball_by_ball_csv_string)
```

## Key Insights

- **Sample size > model complexity**: A pooled model on 339 matches beats 10 per-team models with ~35 matches each
- **Feature engineering has diminishing returns**: Base features capture most signal; additional features add noise
- **Non-linear models don't help**: Random Forest and Gradient Boosting overfit on this sample size
- **~10 RMSE is the prediction floor**: Residual error comes from fundamentally unpredictable T20 events in overs 4-6
- **Simplicity wins**: Unweighted OLS with 5 features outperforms every complex alternative tested

## Data Source

Ball-by-ball data from [Cricsheet](https://cricsheet.org/) in JSON format, covering IPL seasons 2020–2026 (419 first-innings matches).
