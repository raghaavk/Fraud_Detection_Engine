import os
import sys
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from data_loader import load_training_data, engineer_features
import config  # this loads all paths
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..', 'data'))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..', 'features'))


def train_xgboost():
    # ── Load data ─────────────────────────────────────────────────
    df = load_training_data()
    X, y, feature_cols = engineer_features(df)

    # ── Train/test split ──────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📊 Train size: {len(X_train)}, Test size: {len(X_test)}")

    # ── Handle class imbalance with SMOTE ─────────────────────────
    # Fraud is only ~3-5% of data — SMOTE creates synthetic
    # fraud samples so model learns fraud patterns better
    print("\n⚖️  Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"   After SMOTE — fraud: {y_train_bal.sum()}, "
          f"normal: {(y_train_bal == 0).sum()}")

    # ── Train XGBoost ─────────────────────────────────────────────
    print("\n🚀 Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,  # balanced by SMOTE already
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_bal, y_train_bal,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # ── Evaluate ──────────────────────────────────────────────────
    print("\n📈 Evaluation Results:")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    print(f"\n   ROC-AUC Score: {auc:.4f}")
    print(
        f"\n{classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'])}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"   Confusion Matrix:")
    print(f"   True Negatives:  {cm[0][0]}  False Positives: {cm[0][1]}")
    print(f"   False Negatives: {cm[1][0]}  True Positives:  {cm[1][1]}")

    # ── Save model ────────────────────────────────────────────────
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(model, "saved_models/xgboost_fraud.pkl")
    joblib.dump(feature_cols, "saved_models/feature_cols.pkl")
    print("\n✅ XGBoost model saved to ml/saved_models/")

    return model, feature_cols, X_test, y_test


if __name__ == "__main__":
    train_xgboost()
