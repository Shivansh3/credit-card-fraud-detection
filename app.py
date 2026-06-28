"""
Fraud Detection FastAPI Application
Takes only 5 engineered features, automatically fills PCA features
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# INITIALIZE APP
# ============================================================================

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Fraud detection - Input 5 features, get prediction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# LOAD MODEL
# ============================================================================

try:
    model = joblib.load('models/fraud_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    with open('models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    print("✅ Model loaded successfully")
except FileNotFoundError as e:
    raise RuntimeError(f"Model files not found. Run fraud_detection.py first.")

# Extract settings
THRESHOLD_REVIEW = metadata.get('threshold_review', 0.20)
THRESHOLD_BLOCK = metadata.get('threshold_block', 0.70)
FEATURE_NAMES = metadata.get('feature_names', [])

print(f"   Features expected: {len(FEATURE_NAMES)}")
print(f"   User inputs needed: 5 (log_amount, hour_of_day, is_night, amount_zscore, is_small_transaction)")

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class TransactionInput(BaseModel):
    """User provides only these 5 features"""
    log_amount: float
    hour_of_day: float
    is_night: int
    amount_zscore: float
    is_small_transaction: int
    
    class Config:
        schema_extra = {
            "example": {
                "log_amount": 2.5,
                "hour_of_day": 14.5,
                "is_night": 0,
                "amount_zscore": 0.8,
                "is_small_transaction": 0
            }
        }

class BatchInput(BaseModel):
    """Batch of transactions"""
    transactions: List[TransactionInput]

class PredictionResponse(BaseModel):
    """Single prediction"""
    probability: float
    decision: str
    risk_level: str

class BatchPredictionResponse(BaseModel):
    """Batch predictions"""
    predictions: List[PredictionResponse]
    summary: Dict

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_decision(prob: float) -> tuple:
    """Convert probability to decision"""
    if prob >= THRESHOLD_BLOCK:
        return "BLOCK", "HIGH"
    elif prob >= THRESHOLD_REVIEW:
        return "REVIEW", "MEDIUM"
    else:
        return "APPROVE", "LOW"

def prepare_features(transaction: TransactionInput) -> np.ndarray:
    """
    Convert 5 user features to 33 features for model
    
    Model needs: V1-V28 (PCA) + 5 engineered features
    User provides: 5 engineered features
    We fill: V1-V28 with zeros
    """
    
    # Create array with 28 PCA features (V1-V28) as zeros
    pca_features = np.zeros(28)
    
    # Add the 5 engineered features
    user_features = np.array([
        transaction.log_amount,
        transaction.hour_of_day,
        transaction.is_night,
        transaction.amount_zscore,
        transaction.is_small_transaction
    ])
    
    # Combine: V1-V28 (zeros) + 5 engineered
    all_features = np.concatenate([pca_features, user_features])
    
    # Reshape for model
    return all_features.reshape(1, -1)

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
def root():
    """API information"""
    return {
        "name": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "input_features": 5,
        "input_description": {
            "log_amount": "Natural log of transaction amount",
            "hour_of_day": "Hour when transaction occurred (0-24)",
            "is_night": "1 if night transaction, 0 otherwise",
            "amount_zscore": "How unusual the amount is (z-score)",
            "is_small_transaction": "1 if small (<$10), 0 otherwise"
        },
        "thresholds": {
            "approve": f"< {THRESHOLD_REVIEW} (Low risk)",
            "review": f"{THRESHOLD_REVIEW}-{THRESHOLD_BLOCK} (Medium risk)",
            "block": f">= {THRESHOLD_BLOCK} (High risk)"
        }
    }

@app.get("/health", tags=["Status"])
def health_check():
    """Check if API is alive"""
    return {
        "status": "healthy",
        "model": metadata.get('algorithm'),
        "pr_auc": metadata.get('pr_auc_test_set'),
        "roc_auc": metadata.get('roc_auc')
    }

@app.get("/metadata", tags=["Info"])
def get_metadata():
    """Get model performance metrics"""
    return {
        "algorithm": metadata.get('algorithm'),
        "pr_auc_walk_forward": metadata.get('pr_auc_walk_forward'),
        "pr_auc_test_set": metadata.get('pr_auc_test_set'),
        "roc_auc": metadata.get('roc_auc'),
        "f2_score": metadata.get('f2_score'),
        "threshold_review": THRESHOLD_REVIEW,
        "threshold_block": THRESHOLD_BLOCK
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(transaction: TransactionInput):
    """
    Predict fraud for a single transaction
    
    Input 5 features and get fraud probability (0-1)
    """
    try:
        # Prepare features
        features = prepare_features(transaction)
        
        # Scale
        scaled_features = scaler.transform(features)
        
        # Predict
        probability = float(model.predict_proba(scaled_features)[0][1])
        
        # Get decision
        decision, risk_level = get_decision(probability)
        
        return PredictionResponse(
            probability=round(probability, 4),
            decision=decision,
            risk_level=risk_level
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(batch: BatchInput):
    """
    Predict fraud for multiple transactions
    """
    try:
        predictions = []
        decisions_count = {"APPROVE": 0, "REVIEW": 0, "BLOCK": 0}
        
        for transaction in batch.transactions:
            # Prepare and predict
            features = prepare_features(transaction)
            scaled_features = scaler.transform(features)
            probability = float(model.predict_proba(scaled_features)[0][1])
            decision, risk_level = get_decision(probability)
            
            decisions_count[decision] += 1
            predictions.append(
                PredictionResponse(
                    probability=round(probability, 4),
                    decision=decision,
                    risk_level=risk_level
                )
            )
        
        summary = {
            "total": len(batch.transactions),
            "approve": decisions_count["APPROVE"],
            "review": decisions_count["REVIEW"],
            "block": decisions_count["BLOCK"],
            "fraud_rate": round(decisions_count["BLOCK"] / len(batch.transactions), 4)
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
