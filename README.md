# Tanzania Water Pumps — Multiclass Classification

End-to-end machine-learning project for predicting the operating status of water pumps in Tanzania: `functional`, `functional needs repair`, or `non functional`.

## Business context

Reliable pump-status predictions can help public agencies prioritise inspections and maintenance, focusing limited resources on infrastructure most likely to require intervention.

## Approach

- Clean geographic, temporal and operational fields.
- Engineer pump age, population, water-per-person and missingness indicators.
- Frequency-encode high-cardinality categorical variables.
- Compare Random Forest and XGBoost with stratified validation.
- Blend model probabilities and export a DrivenData-compatible submission.

## Result

The strongest competition submission recorded during the project achieved **0.8199 accuracy**. The public version keeps the modelling approach reproducible without including competition data.

## Repository structure

```text
├── src/train.py
├── data/README.md
├── requirements.txt
└── README.md
```

## Run

```bash
python src/train.py \
  --train-values data/raw/train_values.csv \
  --train-labels data/raw/train_labels.csv \
  --test-values data/raw/test_values.csv \
  --output submission.csv
```

## Technologies

Python · Pandas · Scikit-learn · XGBoost · Stratified cross-validation

