from feature_engineering import compute_all_features
from redis_connection import get_redis_client
from db_connection import get_connection
import json
import config
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def warm_up_feature_store(limit=200):
    """
    Precomputes features for `limit` users and stores in Redis.
    In production this would run as a daily cron job.
    """
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    # get recent transactions (one per user)
    cursor.execute("""
        SELECT t.transaction_id, t.user_id, t.amount,
               t.device_fingerprint, t.city,
               t.transaction_hour, t.merchant_category
        FROM transactions t
        INNER JOIN (
            SELECT user_id, MAX(created_at) as latest
            FROM transactions GROUP BY user_id
        ) latest_txns
        ON t.user_id = latest_txns.user_id
        AND t.created_at = latest_txns.latest
        LIMIT %s
    """, (limit,))

    transactions = cursor.fetchall()
    cursor.close()
    conn.close()

    print(f"🔥 Warming up Redis with {len(transactions)} users...")

    success = 0
    for i, txn in enumerate(transactions):
        try:
            compute_all_features(txn)
            success += 1
            if (i + 1) % 50 == 0:
                print(f"   Processed {i + 1}/{len(transactions)}")
        except Exception as e:
            print(f"   ⚠️  Failed for user {txn['user_id']}: {e}")

    print(
        f"\n✅ Feature store warmed up: {success}/{len(transactions)} users cached")

    # show Redis memory usage
    client = get_redis_client()
    info = client.info("memory")
    print(f"📊 Redis memory used: {info['used_memory_human']}")


if __name__ == "__main__":
    warm_up_feature_store(limit=200)
