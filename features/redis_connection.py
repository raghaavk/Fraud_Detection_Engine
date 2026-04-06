from dotenv import load_dotenv
import redis
import config
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()


def get_redis_client():
    """Returns a Redis client connection"""
    try:
        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True  # returns strings instead of bytes
        )
        client.ping()  # test connection
        return client
    except redis.ConnectionError as e:
        print(f"Redis connection failed: {e}")
        raise


def test_redis():
    client = get_redis_client()
    client.set("test_key", "hello_fraud_engine")
    value = client.get("test_key")
    print(f" Redis working! test value: {value}")
    client.delete("test_key")


if __name__ == "__main__":
    test_redis()
