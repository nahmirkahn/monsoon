#!/usr/bin/env python3
import json
from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


ROOT = Path('/home/miso/Documents/WINDOWS/monsoon/final_solution')
TRAIN_CSV = ROOT / 'final_training_dataset.csv'
ULTIMATE_CSV = ROOT / 'ultimate_dataset.csv'
PROJECT_ROOT = Path('/home/miso/Documents/WINDOWS/monsoon')
TRAIN_FLAG_CSV = PROJECT_ROOT / 'senior_ds_test' / 'data' / 'train' / 'train_flag.csv'


def load_best_params() -> dict:
    candidates = [
        ROOT / 'results_lightgbm_optimized.json',
        ROOT / 'final_model_results.json',
        ROOT / 'results_lightgbm.json',
    ]
    best = {}
    for p in candidates:
        if not p.exists():
            continue
        try:
            blob = json.loads(p.read_text())
            for k in ['best_params', 'best_hyperparams', 'params', 'model_params']:
                v = blob.get(k)
                if isinstance(v, dict):
                    best.update(v)
            # Flat fallbacks
            for k in ['num_leaves','n_estimators','max_depth','learning_rate','subsample','colsample_bytree','reg_alpha','reg_lambda','min_child_samples']:
                if k in blob:
                    best[k] = blob[k]
        except Exception:
            pass
    if not best:
        best = {
            'n_estimators': 500,
            'learning_rate': 0.05,
            'max_depth': -1,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'min_child_samples': 20,
            'random_state': 42,
            'n_jobs': -1,
        }
    return best


def detect_target(df: pd.DataFrame) -> str:
    for c in ['target','label','y','default_flag','flag','is_default']:
        if c in df.columns:
            return c
    raise RuntimeError('Target column not found in training CSV')


def main() -> None:
    # Prefer the ultimate dataset (with TARGET) exactly as used by the final script
    if ULTIMATE_CSV.exists():
        print('[train_model_only] Loading ultimate dataset...')
        df = pd.read_csv(ULTIMATE_CSV)
        if 'TARGET' not in df.columns:
            raise RuntimeError('TARGET not found in ultimate_dataset.csv')
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
        model_name = 'lightgbm_final_like'
        print(f'[train_model_only] Training {model_name}...')
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_valid)[:, 1]
        auc = float(roc_auc_score(y_valid, proba))
        print(f'[train_model_only] Validation ROC AUC: {auc:.4f}')
    else:
        if not TRAIN_CSV.exists():
            raise FileNotFoundError(f'Missing {TRAIN_CSV} and {ULTIMATE_CSV}')

        print('[train_model_only] Loading training data...')
        df = pd.read_csv(TRAIN_CSV)
        try:
            target = detect_target(df)
        except RuntimeError:
            if TRAIN_FLAG_CSV.exists():
                print('[train_model_only] Target not found in training CSV, attempting merge with train_flag.csv')
                flag_df = pd.read_csv(TRAIN_FLAG_CSV)
                possible_targets = [c for c in flag_df.columns if c.lower() not in {'id','uuid','customer_id','case_id'}]
                if not possible_targets:
                    raise RuntimeError('No target column found in train_flag.csv')
                target = possible_targets[-1]
                join_key = None
                candidate_keys = ['id','uuid','customer_id','case_id','app_id','account_id','enquiry_id','user_id']
                shared_cols = [c for c in df.columns if c in flag_df.columns]
                candidate_keys += [c for c in shared_cols if re.search(r"_id$", c)]
                for key in candidate_keys:
                    if key in df.columns and key in flag_df.columns:
                        try:
                            u_l = df[key].nunique(dropna=False)
                            u_r = flag_df[key].nunique(dropna=False)
                            if u_l > 0 and u_r > 0:
                                join_key = key
                                break
                        except Exception:
                            continue
                if join_key:
                    df = df.merge(flag_df[[join_key, target]], on=join_key, how='left')
                else:
                    if len(df) == len(flag_df):
                        print('[train_model_only] Falling back to positional merge (row-wise)')
                        df = df.reset_index(drop=True)
                        flag_df = flag_df.reset_index(drop=True)
                        df[target] = flag_df[target]
                    else:
                        raise RuntimeError('Could not find a common join key to merge labels')
                if target not in df.columns:
                    raise RuntimeError('Merge did not add target column')
                print(f'[train_model_only] Merged target `{target}` on key `{join_key}`')
            else:
                raise

        X = df.drop(columns=[target])
        y = df[target]
        X = X.select_dtypes(include=['number','bool'])

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() <= 20 else None
        )

        params = load_best_params()
        print('[train_model_only] Using params:', params)

        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(**params)
            model_name = 'lightgbm'
        except Exception:
            from sklearn.ensemble import RandomForestClassifier
            rf_params = {k: v for k, v in params.items() if k in {'n_estimators','max_depth','random_state','n_jobs'}}
            if not rf_params:
                rf_params = {'n_estimators': 500, 'max_depth': None, 'random_state': 42, 'n_jobs': -1}
            model = RandomForestClassifier(**rf_params)
            model_name = 'random_forest'

        print(f'[train_model_only] Training {model_name}...')
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_valid)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_valid)
        auc = float(roc_auc_score(y_valid, proba))
        print(f'[train_model_only] Validation ROC AUC: {auc:.4f}')

    out = {'model': model_name, 'roc_auc': auc}
    (ROOT / 'retrained_results.json').write_text(json.dumps(out, indent=2))
    print('[train_model_only] Saved retrained_results.json')


if __name__ == '__main__':
    main()


