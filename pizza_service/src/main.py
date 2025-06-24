import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .db import log_prediction
from .model_utils import predict

app = FastAPI(
    title="RAOP Pizza-Success Predictor",
    version="1.0",
    description="Predicts how likely a Reddit pizza request is to succeed.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent.parent / "static"
if not static_dir.exists():
    static_dir.mkdir(parents=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class RequestInput(BaseModel):
    request_title: str
    request_text_edit_aware: str
    request_time_utc: datetime = Field(
        ..., description="ISO timestamp in UTC, e.g. 2025-06-24T11:22:00Z"
    )

    requester_account_age_in_days_at_request: Optional[int] = 0
    requester_upvotes_plus_downvotes_at_request: Optional[int] = 0
    requester_upvotes_minus_downvotes_at_request: Optional[int] = 0
    requester_number_of_posts_on_raop_at_request: Optional[int] = 0
    requester_number_of_posts_at_request: Optional[int] = 0
    requester_number_of_comments_at_request: Optional[int] = 0
    requester_number_of_subreddits_at_request: Optional[int] = 0
    requester_subreddits_at_request: Optional[List[str]] = None


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {
        "message": "Welcome to the RAOP Pizza Predictor API. Visit /docs for API documentation or /app for the web interface."
    }


@app.get("/app")
async def web_app():
    from fastapi.responses import FileResponse

    static_file_path = static_dir / "index.html"
    return FileResponse(static_file_path)


@app.get("/test")
async def test_app():
    from fastapi.responses import FileResponse

    test_file_path = static_dir / "test.html"
    return FileResponse(test_file_path)


@app.post("/predict")
def predict_endpoint(payload: RequestInput):
    try:
        df = pd.DataFrame([payload.model_dump()])
        score, label, feat_df = predict(df)
        log_prediction(score, label, feat_df)
        return {"score": score, "label": label}
    except Exception as exc:
        import traceback

        error_message = str(exc)
        print(f"Error in prediction: {error_message}")
        raise HTTPException(
            status_code=500, detail="An error occurred during prediction"
        )
