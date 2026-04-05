import mysql.connector
from dotenv import load_dotenv
import os

load_dotenv()


def get_connection():
    """Returns a MySQL connection using credentials from .env"""
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST"),
            port=int(os.getenv("DB_PORT")),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
        return connection
    except mysql.connector.Error as e:
        print(f"Database connection failed: {e}")
        raise


def test_connection():
    conn = get_connection()
    if conn.is_connected():
        print(" Connected to MySQL successfully!")
        conn.close()


if __name__ == "__main__":
    test_connection()
