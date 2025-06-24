from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .config import BUCKET_PATTERNS, EMB_MODEL_PATH, REFERENCE_BLOBS, TOP_SUBS

analyzer = SentimentIntensityAnalyzer()

# Text embedding functions


def embed_texts(texts: list[str], model_path: str, batch: int = 32) -> np.ndarray:
    model = SentenceTransformer(model_path, local_files_only=True)
    return model.encode(texts, batch_size=batch, show_progress_bar=False)


# Feature extraction functions


def add_bucket_deciles(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["req_len"] = d.req.str.split().str.len().replace(0, np.nan)
    for name, pat in BUCKET_PATTERNS.items():
        f = d.req.str.count(pat) / d.req_len
        nz = f[f > 0]
        edges = np.quantile(nz, np.linspace(0, 0.9, 10)) if len(nz) else []
        d[name] = np.where(f == 0, 0, np.digitize(f, edges, right=True) + 1)
    return d


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["hour"] = d.request_time_utc.dt.hour
    d["hour_sin"] = np.sin(2 * np.pi * d.hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d.hour / 24)
    d["dow"] = d.request_time_utc.dt.weekday
    d["dow_sin"] = np.sin(2 * np.pi * d.dow / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d.dow / 7)

    days_diff = (d.request_time_utc - d.request_time_utc.min()).dt.days
    if days_diff.nunique() <= 1:
        d["d_comm_age"] = 0
    else:
        d["d_comm_age"] = pd.qcut(days_diff, 10, labels=False, duplicates="drop")

    d["month_h1"] = (d.request_time_utc.dt.day <= 15).astype(int)
    d["log_account_age"] = np.log1p(d.requester_account_age_in_days_at_request)
    return d


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    t = d.req
    d["n_chars"] = t.str.len()
    d["n_words"] = t.str.split().str.len()
    d["cnt_please"] = t.str.count(r"\bplease\b")
    d["cnt_thank"] = t.str.count(r"\bthank\b")
    d["vader_sent"] = t.map(lambda s: analyzer.polarity_scores(s)["compound"])
    d["sentiment_score"] = t.map(lambda s: TextBlob(s).sentiment.polarity)
    pos_med = d.loc[d.sentiment_score > 0, "sentiment_score"].median()
    neg_med = d.loc[d.sentiment_score < 0, "sentiment_score"].median()
    d["sent_pos"] = (d.sentiment_score > pos_med).astype(int)
    d["sent_neg"] = (d.sentiment_score < neg_med).astype(int)
    d["gratitude"] = t.str.contains(r"thank|appreciate").astype(int)
    d["hyperlink"] = t.str.contains(r"http").astype(int)
    d["reciprocity"] = t.str.contains(r"pay.+forward|return.+favor").astype(int)
    return d


def add_social_activity(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    age = d.requester_account_age_in_days_at_request.replace(0, np.nan)
    d["vote_sum"] = d.requester_upvotes_plus_downvotes_at_request
    d["vote_diff"] = d.requester_upvotes_minus_downvotes_at_request
    d["vote_ratio"] = d.vote_diff / d.vote_sum.replace(0, np.nan)
    d["d_karma"] = pd.qcut(d.vote_diff.fillna(0), 10, labels=False, duplicates="drop")
    d["posted_before"] = (d.requester_number_of_posts_on_raop_at_request > 0).astype(
        int
    )
    d["posts_per_day"] = d.requester_number_of_posts_at_request / age
    d["comments_per_day"] = d.requester_number_of_comments_at_request / age
    d["votes_per_day"] = d.requester_upvotes_plus_downvotes_at_request / age
    d["subreddits_per_day"] = d.requester_number_of_subreddits_at_request / age
    return d


def add_subreddit_flags(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    lists = d.requester_subreddits_at_request
    for s in TOP_SUBS:
        d[f"sub_{s}"] = lists.apply(lambda L: 1 if isinstance(L, str) and s in L else 0)
    d["num_top_subs"] = d[[f"sub_{s}" for s in TOP_SUBS]].sum(axis=1)
    return d


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["hour_sentiment"] = d.hour_sin * d.vader_sent
    d["length_upvote"] = d.n_words * d.vote_ratio
    return d


def add_blob_similarity(df: pd.DataFrame, embs: np.ndarray) -> pd.DataFrame:
    mod = SentenceTransformer(str(EMB_MODEL_PATH))
    blob_vecs = mod.encode(list(REFERENCE_BLOBS.values()), show_progress_bar=False)
    sims = (
        embs
        @ blob_vecs.T
        / (
            np.linalg.norm(embs, axis=1, keepdims=True)
            * np.linalg.norm(blob_vecs, axis=1)
        )
    )
    d = df.copy()
    for i, bid in enumerate(REFERENCE_BLOBS):
        d[f"sim_blob_{bid}"] = sims[:, i]
    d["blob_sim_max"] = sims.max(axis=1)
    d["blob_sim_top3_avg"] = np.sort(sims, axis=1)[:, -3:].mean(axis=1)
    d["blob_best_id"] = sims.argmax(axis=1)
    return d


def build_feature_frame(df: pd.DataFrame, embs: np.ndarray) -> pd.DataFrame:
    return (
        df.pipe(add_bucket_deciles)
        .pipe(add_time_features)
        .pipe(add_text_features)
        .pipe(add_social_activity)
        .pipe(add_subreddit_flags)
        .pipe(add_interactions)
        .pipe(add_blob_similarity, embs=embs)
    )
