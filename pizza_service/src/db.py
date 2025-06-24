"""
SQLite database logger for prediction history.
"""

import json
from datetime import datetime
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine

from .config import DB_PATH

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)


class PredictionLog(SQLModel, table=True):
    """Model for storing prediction results."""

    id: Optional[int] = Field(default=None, primary_key=True)
    ts: datetime
    score: float
    label: int
    features_json: str


SQLModel.metadata.create_all(engine)


def log_prediction(score: float, label: int, features_df):
    """Log a prediction to the database with its associated features."""
    # Convert DataFrame to dict
    payload = features_df.to_dict(orient="records")[0]

    # Handle datetime serialization
    for key, value in payload.items():
        if hasattr(value, "isoformat"):
            payload[key] = value.isoformat()

    with Session(engine) as sess:
        rec = PredictionLog(
            ts=datetime.utcnow(),
            score=score,
            label=label,
            features_json=json.dumps(payload),
        )
        sess.add(rec)
        sess.commit()
