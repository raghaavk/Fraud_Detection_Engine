from db_connection import get_connection
import pandas as pd
import numpy as np
import math
import sys
import os

DATA_DIR = r"C:\Users\ragha\OneDrive\Desktop\Fraud- Detection-System\data"
sys.path.insert(0, DATA_DIR)


def load_training_data():
    print("📦 Loading data from MySQL...")
    conn = get_connection()
    query = """
        SELECT
            t.transaction_id, t.user_id, t.amount,
            t.merchant_category, t.device_fingerprint,
            t.ip_address, t.city, t.transaction_hour,
            t.is_new_device, t.is_fraud,
            u.account_age_days, u.avg_transaction_amount,
            u.city as user_home_city
        FROM transactions t
        JOIN users u ON t.user_id = u.user_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    print(f"✅ Loaded {len(df)} transactions")
    print(f"   Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    return df


def engineer_features(df):
    print("\n⚙️  Engineering features...")
    df['amount_log'] = np.log1p(df['amount'])
    df['amount_vs_avg'] = df['amount'] / (df['avg_transaction_amount'] + 1)
    df['amount_z_score'] = (df['amount'] - df['avg_transaction_amount']
                            ) / (df['avg_transaction_amount'].std() + 1)
    df['is_high_amount'] = (
        df['amount'] > df['avg_transaction_amount'] * 3).astype(int)
    df['hour_sin'] = df['transaction_hour'].apply(
        lambda h: math.sin(2 * math.pi * h / 24))
    df['hour_cos'] = df['transaction_hour'].apply(
        lambda h: math.cos(2 * math.pi * h / 24))
    df['is_late_night'] = df['transaction_hour'].apply(
        lambda h: 1 if h in [0, 1, 2, 3, 4] else 0)
    df['is_business_hours'] = df['transaction_hour'].apply(
        lambda h: 1 if 9 <= h <= 18 else 0)
    df['is_new_device'] = df['is_new_device'].astype(int)
    df['is_city_mismatch'] = (df['city'] != df['user_home_city']).astype(int)
    category_risk = {
        'atm': 0.85, 'luxury': 0.75, 'electronics': 0.65,
        'travel': 0.55, 'entertainment': 0.40, 'clothing': 0.25,
        'food_delivery': 0.20, 'grocery': 0.15, 'pharmacy': 0.15, 'fuel': 0.30
    }
    df['category_risk_score'] = df['merchant_category'].map(
        category_risk).fillna(0.5)
    df['is_high_risk_category'] = (
        df['category_risk_score'] >= 0.65).astype(int)
    df['is_new_account'] = (df['account_age_days'] < 90).astype(int)
    df['account_age_log'] = np.log1p(df['account_age_days'])
    df = pd.get_dummies(df, columns=['merchant_category'], prefix='cat')
    feature_cols = [
        'amount', 'amount_log', 'amount_vs_avg', 'amount_z_score',
        'is_high_amount', 'hour_sin', 'hour_cos', 'is_late_night',
        'is_business_hours', 'is_new_device', 'is_city_mismatch',
        'category_risk_score', 'is_high_risk_category',
        'is_new_account', 'account_age_log'
    ] + [c for c in df.columns if c.startswith('cat_')]
    X = df[feature_cols]
    y = df['is_fraud'].astype(int)
    print(f"✅ Feature matrix shape: {X.shape}")
    return X, y, feature_cols


if __name__ == "__main__":
    df = load_training_data()
    X, y, cols = engineer_features(df)
    print(f"\n📊 Class distribution:")
    print(y.value_counts())
