# Drift Monitoring Interpretation

## Setup
- Reference: random 70% of Telco churn data (training-time distribution).
- Current: remaining 30%, with **synthetic drift** injected.

## Engineered perturbations
- `MonthlyCharges`: shifted by ~N(25, 8) noise per row.
- `tenure`: increased by a random integer offset in [6, 18].
- `Contract`: ~45% of current rows forced to `Month-to-month`.
- `Churn`: ~20% of labels flipped to simulate concept/label drift.

## Custom metrics
- Absolute mean difference in `MonthlyCharges` (Evidently custom metric): **24.408**
  (threshold=5.0).
- Absolute churn-rate shift within `Contract == Month-to-month`: **0.0275**.

## Detected drift
- Dataset drift flagged: **True**
- Drifted columns reported: ['Churn', 'Contract', 'MonthlyCharges', 'tenure']
- Injected features that were flagged: ['MonthlyCharges', 'tenure', 'Contract']
- Significant drift (for Airflow): **True**

## Production implication
If this pattern appeared in production, feature and/or target distributions would no longer
match training data. Predictions (especially for monthly-charge-sensitive and month-to-month
segments) would be unreliable. Recommended action: investigate data pipelines, then **retrain**
and re-evaluate before promoting a new Production model.
