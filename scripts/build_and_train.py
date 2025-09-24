#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


ROOT = Path('/home/miso/Documents/WINDOWS/monsoon/final_solution')
ART_CSV = ROOT / 'artifacts' / 'csv'
ART_JSON = ROOT / 'artifacts' / 'json'
ART_PLOTS = ROOT / 'artifacts' / 'plots'

SRC = ROOT / 'data'  # input CSVs location moved here


def ensure_dirs():
    ART_CSV.mkdir(parents=True, exist_ok=True)
    ART_JSON.mkdir(parents=True, exist_ok=True)
    ART_PLOTS.mkdir(parents=True, exist_ok=True)


def recreate_ultimate_dataset() -> Path:
    # For this final solution, we assume ultimate_dataset.csv is available in ROOT.
    # If you need to rebuild from components, plug in your logic here.
    src = SRC / 'ultimate_dataset.csv'
    if not src.exists():
        raise FileNotFoundError(f'Missing {src}. Please place the final ultimate_dataset.csv here.')
    dst = ART_CSV / 'ultimate_dataset.csv'
    if dst.resolve() != src.resolve():
        dst.write_bytes(src.read_bytes())
    return dst


def train_model(ultimate_csv: Path) -> dict:
    df = pd.read_csv(ultimate_csv)
    if 'TARGET' not in df.columns:
        raise RuntimeError('TARGET not found in ultimate dataset')
    feature_cols = [c for c in df.columns if c not in ['uid','NAME_CONTRACT_TYPE','TARGET']]
    X = df[feature_cols].fillna(0)
    y = df['TARGET']

    pos_weight = (y == 0).sum() / max(1, (y == 1).sum())

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        objective='binary', metric='auc', boosting_type='gbdt',
        num_leaves=200, learning_rate=0.05, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=5, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=0.1, n_estimators=1000,
        scale_pos_weight=pos_weight, random_state=42, n_jobs=-1, verbose=-1
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_valid)[:, 1]
    auc = float(roc_auc_score(y_valid, proba))

    out = {
        'auc': auc,
        'num_features': len(feature_cols),
        'class_ratio': float(pos_weight),
    }
    (ART_JSON / 'retrained_results.json').write_text(json.dumps(out, indent=2))

    # Save a lightweight model if needed
    try:
        import joblib
        joblib.dump(model, ART_JSON / 'model_lgbm.pkl')
    except Exception:
        pass

    return {'model': model, 'X_valid': X_valid, 'y_valid': y_valid, 'proba': proba, 'feature_cols': feature_cols}


def main():
    ensure_dirs()
    ultimate_csv = recreate_ultimate_dataset()
    print('[build_and_train] ultimate_dataset at', ultimate_csv)
    bundle = train_model(ultimate_csv)
    print('[build_and_train] AUC:', float(roc_auc_score(bundle['y_valid'], bundle['proba'])))


if __name__ == '__main__':
    main()





