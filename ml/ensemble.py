import os
import sys
import numpy as np
import pandas as pd
import joblib
import shap
import torch
import json
from pytorch_model import FraudDetectionNet
from sklearn.preprocessing import StandardScaler
import config  # this loads all paths
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..', 'data'))


class FraudEnsemble:
    """
    Combines XGBoost + PyTorch predictions with SHAP explainability.
    This is what gets called by the FastAPI scoring endpoint.
    """

    def __init__(self):
        self.xgb_model = None
        self.pytorch_model = None
        self.scaler = None
        self.feature_cols = None
        self.explainer = None
        self._load_models()

    def _load_models(self):
        """Load all saved models from disk"""
        print("🔄 Loading models...")
        models_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'saved_models'
        )

        self.xgb_model = joblib.load(
            os.path.join(models_dir, 'xgboost_fraud.pkl')
        )
        self.scaler = joblib.load(
            os.path.join(models_dir, 'scaler.pkl')
        )
        self.feature_cols = joblib.load(
            os.path.join(models_dir, 'feature_cols.pkl')
        )

        # Load PyTorch model
        input_dim = len(self.feature_cols)
        self.pytorch_model = FraudDetectionNet(input_dim=input_dim)
        self.pytorch_model.load_state_dict(
            torch.load(
                os.path.join(models_dir, 'pytorch_fraud.pth'),
                map_location='cpu'
            )
        )
        self.pytorch_model.eval()

        # SHAP explainer (uses XGBoost as base)
        self.explainer = shap.TreeExplainer(self.xgb_model)
        print("✅ All models loaded")

    def _prepare_features(self, feature_dict: dict) -> np.ndarray:
        """
        Converts a raw feature dict into the numpy array
        the models expect, handling missing columns gracefully.
        """
        row = {}
        for col in self.feature_cols:
            row[col] = feature_dict.get(col, 0)
        return np.array([list(row.values())])

    def predict(self, feature_dict: dict) -> dict:
        """
        Main scoring function.
        Returns risk score, decision, and SHAP explanation.
        """
        X = self._prepare_features(feature_dict)

        # ── XGBoost score ──────────────────────────────────────────
        xgb_score = float(self.xgb_model.predict_proba(X)[0][1])

        # ── PyTorch score ──────────────────────────────────────────
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            pytorch_score = float(self.pytorch_model(X_tensor).item())

        # ── Ensemble: weighted average ─────────────────────────────
        # XGBoost gets 60% weight — it's stronger on tabular data
        # PyTorch gets 40% weight
        ensemble_score = (xgb_score * 0.6) + (pytorch_score * 0.4)

        # ── Decision logic ─────────────────────────────────────────
        if ensemble_score >= 0.75:
            decision = "BLOCK"
        elif ensemble_score >= 0.45:
            decision = "FLAG"
        else:
            decision = "ALLOW"

        # ── SHAP explanation ───────────────────────────────────────
        shap_values = self.explainer.shap_values(X)
        shap_array = shap_values[0] if isinstance(
            shap_values, list
        ) else shap_values[0]

        # top 5 features driving this decision
        feature_impacts = sorted(
            zip(self.feature_cols, shap_array),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        explanation = [
            {
                "feature": feat,
                "impact": round(float(val), 4),
                "direction": "increases_fraud_risk" if val > 0
                             else "decreases_fraud_risk"
            }
            for feat, val in feature_impacts
        ]

        return {
            "ensemble_score": round(ensemble_score, 4),
            "xgb_score": round(xgb_score, 4),
            "pytorch_score": round(pytorch_score, 4),
            "decision": decision,
            "explanation": explanation
        }


# ── Test the ensemble ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    ensemble = FraudEnsemble()

    # simulate a suspicious transaction
    test_features = {
        "amount": 45000,
        "amount_log": 10.7,
        "amount_vs_avg": 8.5,
        "amount_z_score": 4.2,
        "is_high_amount": 1,
        "hour_sin": -0.866,
        "hour_cos": -0.5,
        "is_late_night": 1,
        "is_business_hours": 0,
        "is_new_device": 1,
        "is_city_mismatch": 1,
        "category_risk_score": 0.85,
        "is_high_risk_category": 1,
        "is_new_account": 0,
        "account_age_log": 6.2,
        "cat_atm": 1
    }

    result = ensemble.predict(test_features)

    print("\n🎯 Ensemble Prediction:")
    print(f"   Decision:        {result['decision']}")
    print(f"   Ensemble Score:  {result['ensemble_score']}")
    print(f"   XGBoost Score:   {result['xgb_score']}")
    print(f"   PyTorch Score:   {result['pytorch_score']}")
    print(f"\n🔍 Top reasons for this decision:")
    for exp in result['explanation']:
        arrow = "🔴" if exp['direction'] == 'increases_fraud_risk' else "🟢"
        print(f"   {arrow} {exp['feature']:<30} impact: {exp['impact']}")
