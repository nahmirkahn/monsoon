# Final Solution

## How to Run

1) Activate env (micromamba):
```
micromamba activate mon
```
2) Execute pipeline (train + plots + SHAP):
```
./run.sh
```

## Outputs

- `artifacts/csv/ultimate_dataset.csv` – input to training
- `artifacts/json/retrained_results.json` – metrics (AUC)
- `artifacts/json/model_lgbm.pkl` – trained model for reuse
- `artifacts/json/metrics_summary.json` – AUC, AP, accuracy, precision, recall
- `artifacts/plots/*.png` – ROC, PR, confusion matrix, feature importance
- `artifacts/plots/shap/*.png` – SHAP beeswarm and bar plots

## Scripts

- `scripts/build_and_train.py` – copy ultimate CSV into artifacts and train model
- `scripts/plot_all.py` – generate and display plots; saves to artifacts/plots
- `scripts/shap_analysis.py` – generate and display SHAP plots
- `scripts/plot_viewer.py` – list/show saved plots on demand

## Documentation (copied from solution)

Located under `docs/`:
- `ADVANCED_FEATURE_ENGINEERING_GUIDE.md`
- `COMPLETE_PIPELINE_RESULTS_REPORT.md`
- `COMPREHENSIVE_MODEL_PERFORMANCE_REPORT.md`
- `FEATURE_ENGINEERING_GUIDE.md`
- `FINAL_ANALYSIS_AND_RECOMMENDATIONS.md`



