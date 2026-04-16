from redis_connection import get_redis_client
from db_connection import get_connection
from scoring_service import scoring_service
from models import TransactionRequest, ScoringResponse, HealthResponse
from loguru import logger
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request
from datetime import datetime
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = r"C:\Users\ragha\OneDrive\Desktop\Fraud- Detection-System"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data"))
sys.path.insert(0, os.path.join(ROOT, "features"))
sys.path.insert(0, os.path.join(ROOT, "ml"))


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time transaction fraud scoring engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ── Startup event ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting Fraud Detection API...")
    scoring_service.initialize()
    logger.info("✅ API ready")


# ── Health check endpoint ─────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    # check Redis
    try:
        redis = get_redis_client()
        redis.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    # check MySQL
    try:
        conn = get_connection()
        conn.close()
        mysql_ok = True
    except Exception:
        mysql_ok = False

    return HealthResponse(
        status="healthy" if redis_ok and mysql_ok else "degraded",
        models_loaded=scoring_service._initialized,
        redis_connected=redis_ok,
        mysql_connected=mysql_ok
    )


# ── Main scoring endpoint ─────────────────────────────────────────────────────

@app.post("/score-transaction", response_model=ScoringResponse)
async def score_transaction(request: TransactionRequest):
    start_time = time.time()

    try:
        logger.info(
            f"Scoring transaction {request.transaction_id} "
            f"| amount: {request.amount} "
            f"| user: {request.user_id}"
        )

        # convert pydantic model to dict
        txn_dict = request.dict()

        # get ML score
        result = scoring_service.score(txn_dict)

        processing_ms = round((time.time() - start_time) * 1000, 2)

        # log to audit table
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_log
                (transaction_id, action_taken, risk_score, flagged_reason)
                VALUES (%s, %s, %s, %s)
            """, (
                request.transaction_id,
                result["decision"],
                result["ensemble_score"],
                str([e["feature"] for e in result["explanation"][:2]])
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Audit log failed: {e}")

        logger.info(
            f"Decision: {result['decision']} "
            f"| Score: {result['ensemble_score']} "
            f"| {processing_ms}ms"
        )

        return ScoringResponse(
            transaction_id=request.transaction_id,
            decision=result["decision"],
            ensemble_score=result["ensemble_score"],
            xgb_score=result["xgb_score"],
            pytorch_score=result["pytorch_score"],
            explanation=result["explanation"],
            processing_time_ms=processing_ms,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"Scoring failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Stats endpoint ────────────────────────────────────────────────────────────

@app.get("/stats")
async def get_stats():
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT
                action_taken as decision,
                COUNT(*) as count,
                AVG(risk_score) as avg_score
            FROM audit_log
            GROUP BY action_taken
        """)
        stats = cursor.fetchall()
        conn.close()
        return {"stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
