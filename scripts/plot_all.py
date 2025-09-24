#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, average_precision_score, roc_auc_score


ROOT = Path('/home/miso/Documents/WINDOWS/monsoon/final_solution')
ART_CSV = ROOT / 'artifacts' / 'csv'
ART_JSON = ROOT / 'artifacts' / 'json'
ART_PLOTS = ROOT / 'artifacts' / 'plots'


def load_validation_bundle():
    # Recompute from artifacts CSV
    ulti = ART_CSV / 'ultimate_dataset.csv'
    df = pd.read_csv(ulti)
    feature_cols = [c for c in df.columns if c not in ['uid','NAME_CONTRACT_TYPE','TARGET']]
    X = df[feature_cols].fillna(0)
    y = df['TARGET']
    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Load model if present; else fit quickly to generate plots
    model_path = ART_JSON / 'model_lgbm.pkl'
    model = None
    if model_path.exists():
        import joblib
        model = joblib.load(model_path)
        proba = model.predict_proba(X_valid)[:, 1]
    else:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=300, random_state=42)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_valid)[:, 1]
    return X_valid, y_valid, proba, model, feature_cols


def save_and_show_plots():
    ART_PLOTS.mkdir(parents=True, exist_ok=True)
    X_valid, y_valid, proba, model, feature_cols = load_validation_bundle()

    fpr, tpr, _ = roc_curve(y_valid, proba)
    ap = average_precision_score(y_valid, proba)
    auc = roc_auc_score(y_valid, proba)

    # ROC
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.legend()
    path = ART_PLOTS / 'roc_curve.png'
    plt.savefig(path, bbox_inches='tight'); plt.show()

    # PR
    prec, rec, _ = precision_recall_curve(y_valid, proba)
    plt.figure(figsize=(6,5))
    plt.plot(rec, prec, label=f'AP={ap:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve'); plt.legend()
    path = ART_PLOTS / 'precision_recall_curve.png'
    plt.savefig(path, bbox_inches='tight'); plt.show()

    # Confusion Matrix at 0.5
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_valid, pred)
    acc = (cm[0,0] + cm[1,1]) / cm.sum()
    prec_val = cm[1,1] / max(1, (cm[0,1] + cm[1,1]))
    rec_val = cm[1,1] / max(1, (cm[1,0] + cm[1,1]))
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'CM (thr=0.5) | Acc={acc:.3f} P={prec_val:.3f} R={rec_val:.3f}')
    path = ART_PLOTS / 'confusion_matrix.png'
    plt.savefig(path, bbox_inches='tight'); plt.show()

    # Feature importances
    if hasattr(model, 'feature_importances_'):
        order = np.argsort(model.feature_importances_)[::-1][:30]
        plt.figure(figsize=(8, max(4, len(order)*0.25)))
        sns.barplot(x=model.feature_importances_[order], y=np.array(feature_cols)[order])
        plt.title('Top Feature Importances')
        path = ART_PLOTS / 'feature_importance.png'
        plt.savefig(path, bbox_inches='tight'); plt.show()

    # Final metrics pane (numbers only)
    metrics = {
        'roc_auc': float(auc),
        'average_precision': float(ap),
        'accuracy': float(acc),
        'precision': float(prec_val),
        'recall': float(rec_val),
    }
    print('[plot_all] Metrics summary:')
    for k, v in metrics.items():
        print(f'- {k}: {v:.4f}')
    (ART_JSON / 'metrics_summary.json').write_text(json.dumps(metrics, indent=2))


def main():
    save_and_show_plots()


if __name__ == '__main__':
    main()





