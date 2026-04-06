
from redis_connection import get_redis_client
from db_connection import get_connection
from dotenv import load_dotenv
from datetime import datetime
import hashlib
import json
import config
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()


TTL = int(os.getenv("REDIS_TTL", 86400))


# ─── Helper: make a safe Redis key ────────────────────────────────────────────

def make_key(prefix, user_id):
    """Creates a clean Redis key like features:user:abc123"""
    return f"features:user:{prefix}:{user_id}"


# ─── Feature 1: Velocity Features ─────────────────────────────────────────────
# How many transactions has this user made recently?
# High velocity = suspicious behaviour

def compute_velocity_features(user_id, conn):
    """
    Computes transaction counts in last 1hr, 6hr, 24hr windows.
    Also computes total spend in each window.
    """
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT
            COUNT(*) as total_txns,

            SUM(CASE WHEN created_at >= NOW() - INTERVAL 1 HOUR
                THEN 1 ELSE 0 END) as txns_last_1hr,

            SUM(CASE WHEN created_at >= NOW() - INTERVAL 6 HOUR
                THEN 1 ELSE 0 END) as txns_last_6hr,

            SUM(CASE WHEN created_at >= NOW() - INTERVAL 24 HOUR
                THEN 1 ELSE 0 END) as txns_last_24hr,

            SUM(CASE WHEN created_at >= NOW() - INTERVAL 1 HOUR
                THEN amount ELSE 0 END) as spend_last_1hr,

            SUM(CASE WHEN created_at >= NOW() - INTERVAL 24 HOUR
                THEN amount ELSE 0 END) as spend_last_24hr,

            AVG(amount) as avg_amount_alltime,
            MAX(amount) as max_amount_alltime

        FROM transactions
        WHERE user_id = %s
    """, (user_id,))

    row = cursor.fetchone()
    cursor.close()

    return {
        "total_txns": int(row["total_txns"] or 0),
        "txns_last_1hr": int(row["txns_last_1hr"] or 0),
        "txns_last_6hr": int(row["txns_last_6hr"] or 0),
        "txns_last_24hr": int(row["txns_last_24hr"] or 0),
        "spend_last_1hr": float(row["spend_last_1hr"] or 0),
        "spend_last_24hr": float(row["spend_last_24hr"] or 0),
        "avg_amount_alltime": float(row["avg_amount_alltime"] or 0),
        "max_amount_alltime": float(row["max_amount_alltime"] or 0),
    }


# ─── Feature 2: Amount Anomaly Score ──────────────────────────────────────────
# Is this transaction amount unusual for this specific user?

def compute_amount_anomaly(user_id, current_amount, conn):
    """
    Z-score of current transaction vs user's historical amounts.
    Z-score > 3 means very unusual.
    """
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT
            AVG(amount) as mean_amount,
            STD(amount) as std_amount,
            COUNT(*) as history_count
        FROM transactions
        WHERE user_id = %s
    """, (user_id,))

    row = cursor.fetchone()
    cursor.close()

    mean = float(row["mean_amount"] or 0)
    std = float(row["std_amount"] or 1)
    count = int(row["history_count"] or 0)

    # avoid division by zero
    if std == 0:
        std = 1

    z_score = (current_amount - mean) / std

    return {
        "amount_z_score": round(z_score, 4),
        "user_mean_amount": round(mean, 2),
        "user_std_amount": round(std, 2),
        "user_history_count": count,
        "is_amount_anomaly": bool(abs(z_score) > 3)
    }


# ─── Feature 3: Device Features ───────────────────────────────────────────────
# Has this user used this device before?

def compute_device_features(user_id, device_fingerprint, conn):
    """
    Checks if current device is new for this user.
    Also counts unique devices used historically.
    """
    cursor = conn.cursor(dictionary=True)

    # get all devices this user has used before
    cursor.execute("""
        SELECT DISTINCT device_fingerprint, COUNT(*) as usage_count
        FROM transactions
        WHERE user_id = %s
        GROUP BY device_fingerprint
        ORDER BY usage_count DESC
    """, (user_id,))

    rows = cursor.fetchall()
    cursor.close()

    known_devices = [r["device_fingerprint"] for r in rows]
    unique_device_count = len(known_devices)
    is_new_device = device_fingerprint not in known_devices

    # primary device = most used device
    primary_device = known_devices[0] if known_devices else None
    is_primary_device = (device_fingerprint == primary_device)

    return {
        "is_new_device": bool(is_new_device),
        "is_primary_device": bool(is_primary_device),
        "unique_device_count": unique_device_count,
        "device_fingerprint_hash": hashlib.md5(
            device_fingerprint.encode()
        ).hexdigest()[:16]
    }


# ─── Feature 4: Location Features ─────────────────────────────────────────────
# Is this transaction happening from an unusual location?

def compute_location_features(user_id, current_city, conn):
    """
    Compares current city to user's most frequent transaction city.
    """
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT city, COUNT(*) as visit_count
        FROM transactions
        WHERE user_id = %s
        GROUP BY city
        ORDER BY visit_count DESC
        LIMIT 1
    """, (user_id,))

    row = cursor.fetchone()
    cursor.close()

    home_city = row["city"] if row else None
    is_city_mismatch = (home_city is not None and current_city != home_city)

    return {
        "home_city": home_city,
        "current_city": current_city,
        "is_city_mismatch": bool(is_city_mismatch)
    }


# ─── Feature 5: Time Features ─────────────────────────────────────────────────
# Is this transaction at an unusual time?

def compute_time_features(transaction_hour):
    """
    Flags late-night transactions and encodes time cyclically
    so ML model understands hour 23 and hour 0 are close.
    """
    import math

    is_late_night = bool(transaction_hour in [0, 1, 2, 3, 4])
    is_business_hours = bool(9 <= transaction_hour <= 18)

    # Cyclic encoding so 23:00 and 00:00 are numerically close
    hour_sin = round(math.sin(2 * math.pi * transaction_hour / 24), 4)
    hour_cos = round(math.cos(2 * math.pi * transaction_hour / 24), 4)

    return {
        "transaction_hour": transaction_hour,
        "is_late_night": is_late_night,
        "is_business_hours": is_business_hours,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos
    }


# ─── Feature 6: Merchant Risk Features ────────────────────────────────────────

def compute_merchant_features(merchant_category, amount):
    """
    Assigns a risk score to merchant categories based on
    historical fraud patterns in FinTech data.
    """
    # higher score = riskier category
    category_risk = {
        'atm': 0.85,
        'luxury': 0.75,
        'electronics': 0.65,
        'travel': 0.55,
        'entertainment': 0.40,
        'food_delivery': 0.20,
        'grocery': 0.15,
        'pharmacy': 0.15,
        'clothing': 0.25,
        'fuel': 0.30
    }

    risk_score = category_risk.get(merchant_category, 0.50)
    is_high_risk_category = bool(risk_score >= 0.65)

    return {
        "merchant_category": merchant_category,
        "category_risk_score": risk_score,
        "is_high_risk_category": is_high_risk_category,
        "amount_category_ratio": round(
            amount / (risk_score * 10000 + 1), 4
        )
    }


# ─── Master Function: Compute All Features ────────────────────────────────────

def compute_all_features(transaction: dict) -> dict:
    """
    Takes a raw transaction dict, computes all features,
    caches them in Redis, and returns the full feature vector.
    """
    user_id = transaction["user_id"]
    amount = float(transaction["amount"])
    device = transaction["device_fingerprint"]
    city = transaction["city"]
    hour = int(transaction["transaction_hour"])
    category = transaction["merchant_category"]

    redis_key = make_key("all_features", user_id)
    redis_client = get_redis_client()

    # Check Redis cache first
    cached = redis_client.get(redis_key)
    if cached:
        features = json.loads(cached)
        # update dynamic fields that change per transaction
        features.update(compute_time_features(hour))
        features.update(compute_merchant_features(category, amount))
        features["amount"] = amount
        features["cache_hit"] = True
        return features

    # Cache miss — compute everything from MySQL
    conn = get_connection()

    features = {}
    features["user_id"] = user_id
    features["transaction_id"] = transaction.get("transaction_id", "")
    features["amount"] = amount
    features["cache_hit"] = False

    features.update(compute_velocity_features(user_id, conn))
    features.update(compute_amount_anomaly(user_id, amount, conn))
    features.update(compute_device_features(user_id, device, conn))
    features.update(compute_location_features(user_id, city, conn))
    features.update(compute_time_features(hour))
    features.update(compute_merchant_features(category, amount))

    conn.close()

    # Store in Redis (cache for 24 hours)
    redis_client.setex(redis_key, TTL, json.dumps(features))

    return features


# ─── Test it ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
    from db_connection import get_connection

    # grab a real user_id from your database
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT t.transaction_id, t.user_id, t.amount,
               t.device_fingerprint, t.city,
               t.transaction_hour, t.merchant_category
        FROM transactions t LIMIT 1
    """)
    sample_txn = cursor.fetchone()
    cursor.close()
    conn.close()

    print(" Sample transaction:")
    print(json.dumps(sample_txn, indent=2, default=str))

    print("\n  Computing features...")
    features = compute_all_features(sample_txn)

    print("\n Feature vector:")
    for key, value in features.items():
        print(f"   {key:<35} {value}")
