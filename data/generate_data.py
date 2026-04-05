import uuid
import random
from faker import Faker
from datetime import datetime, timedelta
import mysql.connector
from db_connection import get_connection

fake = Faker('en_IN')  # Indian locale for realistic FinTech data
random.seed(42)

MERCHANT_CATEGORIES = [
    'grocery', 'electronics', 'food_delivery', 'travel',
    'entertainment', 'clothing', 'pharmacy', 'fuel', 'atm', 'luxury'
]

CITIES = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad',
          'Pune', 'Kolkata', 'Ahmedabad', 'Jaipur', 'Surat']

DEVICES = [f"device_{i}" for i in range(1, 500)]  # pool of 500 devices


# ─── Step 1: Generate Users ───────────────────────────────────────────────────

def generate_users(n=1000):
    """Generate n realistic users"""
    fake.unique.clear()

    users = []
    for _ in range(n):
        users.append({
            'user_id': str(uuid.uuid4()),
            'name': fake.name(),
            'email': fake.unique.email(),
            'phone': fake.phone_number()[:20],
            'account_age_days': random.randint(30, 3000),
            'avg_transaction_amount': round(random.uniform(200, 15000), 2),
            'city': random.choice(CITIES)
        })
    return users


# ─── Step 2: Fraud Labelling Logic ────────────────────────────────────────────

def label_fraud(txn, user):
    """
    Rule-based fraud labelling.
    Returns (is_fraud: bool, reason: str)
    """
    reasons = []

    # Rule 1: Very large amount compared to user's average
    if txn['amount'] > user['avg_transaction_amount'] * 5:
        reasons.append("amount_spike")

    # Rule 2: Transaction at unusual hours (1am - 4am)
    if txn['transaction_hour'] in [1, 2, 3, 4]:
        reasons.append("unusual_hour")

    # Rule 3: New device + large amount
    if txn['is_new_device'] and txn['amount'] > 10000:
        reasons.append("new_device_high_amount")

    # Rule 4: City mismatch (user's city vs transaction city)
    if txn['city'] != user['city'] and txn['amount'] > 5000:
        reasons.append("city_mismatch")

    # Rule 5: High-risk category + new device
    if txn['merchant_category'] in ['atm', 'luxury'] and txn['is_new_device']:
        reasons.append("risky_category_new_device")

    # Label as fraud only if 2+ rules trigger (reduces false positives)
    is_fraud = len(reasons) >= 2
    return is_fraud, ", ".join(reasons) if is_fraud else ""


# ─── Step 3: Generate Transactions ────────────────────────────────────────────

def generate_transactions(users, n=100000):
    """Generate n transactions linked to users"""
    transactions = []
    user_device_map = {}  # tracks which devices each user has used

    for i in range(n):
        user = random.choice(users)
        uid = user['user_id']

        # Build or reuse device history for this user
        if uid not in user_device_map:
            # assign 1-3 known devices to this user
            user_device_map[uid] = random.sample(
                DEVICES, k=random.randint(1, 3))

        # 15% chance transaction comes from a new/unknown device
        is_new_device = random.random() < 0.15
        if is_new_device:
            # outside known pool
            device = f"device_{random.randint(500, 1000)}"
        else:
            device = random.choice(user_device_map[uid])

        category = random.choice(MERCHANT_CATEGORIES)
        hour = random.randint(0, 23)
        city = random.choice(CITIES)

        # Amount logic: luxury/atm transactions tend to be higher
        if category in ['luxury', 'atm']:
            amount = round(random.uniform(5000, 80000), 2)
        elif category == 'electronics':
            amount = round(random.uniform(1000, 50000), 2)
        else:
            amount = round(random.uniform(50, 20000), 2)

        txn = {
            'transaction_id': str(uuid.uuid4()),
            'user_id': uid,
            'amount': amount,
            'merchant_category': category,
            'merchant_name': fake.company()[:100],
            'device_fingerprint': device,
            'ip_address': fake.ipv4(),
            'city': city,
            'country': 'India',
            'transaction_hour': hour,
            'is_new_device': is_new_device,
        }

        is_fraud, reason = label_fraud(txn, user)
        txn['is_fraud'] = is_fraud
        txn['fraud_reason'] = reason

        transactions.append(txn)

        if (i + 1) % 10000 == 0:
            print(f"  Generated {i+1} transactions...")

    return transactions


# ─── Step 4: Insert into MySQL ────────────────────────────────────────────────

def insert_users(conn, users):
    cursor = conn.cursor()

    sql = """
        INSERT INTO users
        (user_id, name, email, phone, account_age_days,avg_transaction_amount, city)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    data = [(u['user_id'], u['name'], u['email'], u['phone'],
             u['account_age_days'], u['avg_transaction_amount'], u['city'])
            for u in users]

    cursor.executemany(sql, data)
    conn.commit()
    print(f" Inserted {len(users)} users")
    cursor.close()


def insert_transactions(conn, transactions, batch_size=5000):
    cursor = conn.cursor()
    sql = """
        INSERT INTO transactions
        (transaction_id, user_id, amount, merchant_category, merchant_name,
         device_fingerprint, ip_address, city, country, transaction_hour,
         is_new_device, is_fraud, fraud_reason)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    # Insert in batches so MySQL doesn't choke
    for i in range(0, len(transactions), batch_size):
        batch = transactions[i:i + batch_size]
        data = [(t['transaction_id'], t['user_id'], t['amount'],
                 t['merchant_category'], t['merchant_name'],
                 t['device_fingerprint'], t['ip_address'],
                 t['city'], t['country'], t['transaction_hour'],
                 t['is_new_device'], t['is_fraud'], t['fraud_reason'])
                for t in batch]
        cursor.executemany(sql, data)
        conn.commit()
        print(f"  Inserted batch up to {i + len(batch)} transactions")

    print(f" Inserted {len(transactions)} transactions total")
    cursor.close()


# ─── Step 5: Verify & Print Stats ─────────────────────────────────────────────

def print_stats(conn):
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM users")
    print(f"\n📊 Total users: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM transactions")
    print(f" Total transactions: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = TRUE")
    fraud_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transactions")
    total = cursor.fetchone()[0]
    print(
        f" Fraud transactions: {fraud_count} ({round(fraud_count/total*100, 2)}%)")

    cursor.execute("""
        SELECT merchant_category, COUNT(*) as cnt, SUM(is_fraud) as frauds
        FROM transactions GROUP BY merchant_category ORDER BY frauds DESC
    """)
    print("\n Fraud by category:")
    for row in cursor.fetchall():
        print(f"   {row[0]:<20} total: {row[1]:<8} frauds: {row[2]}")

    cursor.close()


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(" Starting data generation...")

    print("\n[1/4] Generating users...")
    users = generate_users(n=1000)

    print("\n[2/4] Generating transactions...")
    transactions = generate_transactions(users, n=100000)

    print("\n[3/4] Connecting to database...")
    conn = get_connection()

    print("\n[4/4] Inserting into MySQL...")
    insert_users(conn, users)
    insert_transactions(conn, transactions)

    print_stats(conn)
    conn.close()

    print("\n Phase 1 complete! Database is ready.")
