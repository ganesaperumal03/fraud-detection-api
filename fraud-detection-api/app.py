import joblib
import pandas as pd
from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Load the saved model once, when the API starts
bundle = joblib.load("model/fraud_model.joblib")
model = bundle["model"]
features = bundle["features"]
threshold = bundle["threshold"]

app = FastAPI(title="Fraud Detection API")


# The shape of one request, with rules for each field
class Transaction(BaseModel):
    step: int = Field(ge=1)
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(gt=0)
    oldbalanceOrg: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)


# Turn one request into the same columns the model was trained on
def make_features(t: Transaction) -> pd.DataFrame:
    row = {
        "amount": t.amount,
        "oldbalanceOrg": t.oldbalanceOrg,
        "oldbalanceDest": t.oldbalanceDest,
        "hour": t.step % 24,
        "amount_to_balance": t.amount / (t.oldbalanceOrg + 1),
        "empties_account": int(t.amount >= t.oldbalanceOrg and t.oldbalanceOrg > 0),
    }
    for name in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        row["type_" + name] = int(t.type == name)
    return pd.DataFrame([row])[features]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(t: Transaction):
    X = make_features(t)
    score = float(model.predict_proba(X)[0, 1])
    return {
        "fraud_score": round(score, 4),
        "flagged": score >= threshold,
        "threshold": threshold,
    }
