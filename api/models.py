from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class TransactionRequest(BaseModel):
    transaction_id: str
    user_id: str
    amount: float = Field(..., gt=0, description="Transaction amount in INR")
    merchant_category: str
    merchant_name: str
    device_fingerprint: str
    ip_address: str
    city: str
    country: str = "India"
    transaction_hour: int = Field(..., ge=0, le=23)

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "txn_001",
                "user_id": "usr_001",
                "amount": 45000.00,
                "merchant_category": "atm",
                "merchant_name": "HDFC ATM",
                "device_fingerprint": "device_999",
                "ip_address": "192.168.1.1",
                "city": "Mumbai",
                "country": "India",
                "transaction_hour": 2
            }
        }


class ExplanationItem(BaseModel):
    feature: str
    impact: float
    direction: str


class ScoringResponse(BaseModel):
    transaction_id: str
    decision: str
    ensemble_score: float
    xgb_score: float
    pytorch_score: float
    explanation: List[ExplanationItem]
    processing_time_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    redis_connected: bool
    mysql_connected: bool
    version: str = "1.0.0"
