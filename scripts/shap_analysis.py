#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/home/miso/Documents/WINDOWS/monsoon/final_solution')
ART_CSV = ROOT / 'artifacts' / 'csv'
ART_JSON = ROOT / 'artifacts' / 'json'
ART_PLOTS = ROOT / 'artifacts' / 'plots'
ART_SHAP = ART_PLOTS / 'shap'


def run_shap():
    ART_SHAP.mkdir(parents=True, exist_ok=True)
    import joblib
    import shap
    import matplotlib.pyplot as plt

    model_path = ART_JSON / 'model_lgbm.pkl'
    if not model_path.exists():
        raise FileNotFoundError('Trained model not found. Run build_and_train.py first.')
    model = joblib.load(model_path)

    df = pd.read_csv(ART_CSV / 'ultimate_dataset.csv')
    feature_cols = [c for c in df.columns if c not in ['uid','NAME_CONTRACT_TYPE','TARGET']]
    X = df[feature_cols].fillna(0)

    # sample for speed
    sample = X.sample(n=min(5000, len(X)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(sample)

    plt.figure()
    shap.plots.beeswarm(shap_values, show=False, max_display=25)
    path = ART_SHAP / 'shap_beeswarm.png'
    plt.savefig(path, bbox_inches='tight'); plt.show()

    plt.figure()
    shap.plots.bar(shap_values, show=False, max_display=25)
    path = ART_SHAP / 'shap_bar.png'
    plt.savefig(path, bbox_inches='tight'); plt.show()


def main():
    run_shap()


if __name__ == '__main__':
    main()





