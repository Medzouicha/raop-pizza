from __future__ import annotations

import pickle
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

RND = 42
CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=RND)
# ROOT_DIR        = r"C:\Users\mzouicha\OneDrive - Amadeus Workplace\Desktop\STAGE\raop-pizza"
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "processed" / "dataset_clean.csv"
EMB_MODEL_PATH = ROOT_DIR / "all-MiniLM-L6-v2"  # HF folder or model id
EMB_CACHE = ROOT_DIR / "embeddings_all-MiniLM-L6-v2.npy"

MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
STACK_PATH = MODEL_DIR / "pizza_stacked_ensemble.pkl"
META_PATH = MODEL_DIR / "feature_metadata.pkl"

# narrative buckets & other constants
BUCKET_PATTERNS = {
    "b_family": r"\b(kid(?:s)?|child(?:ren)?|family|mom|dad|wife|husband|pregnant)\b",
    "b_student": r"\b(college|university|student|school|exam|finals?)\b",
    "b_job_loss": r"\b(unemployed|lost\s+my\s+job|jobless|fired)\b",
    "b_broke": r"\b(broke|no\s+money|can't\s+afford)\b",
    "b_payday_gap": r"\b(payday|waiting\s+for\s+paycheck)\b",
    "b_urgent_hunger": r"\b(hungry|starving|no\s+food|empty\s+fridge)\b",
    "b_emotional": r"\b(rock\s+bottom|desperate|panic|depressed)\b",
    "b_pay_it_forward": r"\b(pay\s+it\s+forward|return\s+the\s+favor|repay)\b",
}

TOP_SUBS = [
    "AskReddit",
    "pics",
    "todayilearned",
    "funny",
    "IAmA",
    "WTF",
    "videos",
    "Random_Acts_Of_Pizza",
]

REFERENCE_BLOBS = {
    0: "Hello everyone—today is my birthday, and I’ve never felt more at a loss. I’m a college student who just paid tuition and rent, currently unemployed, with only a few dollars left in my bank account. Normally I’d treat myself to a slice or two to celebrate, but this year I simply can’t afford even that small comfort. My birthday has always been a reminder to be grateful, but right now it feels like just another tough day. If anyone could help me out with a pizza, it would lift my spirits in a way I truly need—and I promise to pay your kindness forward when I’m back on solid ground. Thank you from the bottom of my heart for any help you can offer",
    1: "Cluster theme: college musicians who’ve literally run out of food and money, offering to write and record songs in exchange for pizza.I’m a full-time university student and musician who has just hit $0.00 on my meal card and finished the last of my Easy Mac and sandwiches. With finals next week and no groceries left, I need to survive until payday in seven days.If you send me a pizza today, I will: 1. Send you a *demo MP3 link* of a fully-produced song or jingle—in any genre (folk, orchestral, jazz, electronic, rock). 2. Share my SoundCloud/YouTube samples (3,000+ subscribers) so you know exactly what you’ll get—before you order. 3. Publicly credit you with a shout-out in my next video or Kickstarter update. 4. Pay it forward by composing another custom track for someone else as soon as I get paid.Thank you so much for considering this—I truly appreciate any help right now. Your kindness will keep me fed and inspired through finals week.",
    2: "Cluster theme: genuine pay-it-forward pizza asks from people truly in need.I’m completely out of funds and struggling today—this is my very first request and I’ve hit $0.00 in my account.  If you can help me with a pizza right now, I would appreciate it more than you can imagine.  I promise to pay your kindness forward as soon as I’m back on my feet:   • I will send you a pizza offer or return the favor on Friday when I have money again.   • I’ll provide any proof or verification you need to confirm delivery.  Thank you so much for considering this—your help means everything, and I will pay it forward the moment I’m able.",
    3: "Cluster theme: urgent pizza (“pie”) requests from people who are truly out of food and money.I’m completely out of groceries and cash—my pantry is empty and I’ve eaten my last meal. I’m starving right now and need to hold on until payday in two–three days. If you can spare me a pizza today, I will: 1. Truly appreciate it and send heartfelt thanks immediately. 2. Pay your kindness forward with a pizza gift to another Redditor on payday. 3. Share a photo or confirmation code on request so you know it arrived.Thank you so much for helping me and keeping me fed when I’m at my lowest. Your generosity means the world.",
    4: "Cluster theme: artists and crafters offering specific creative trades (drawing, Photoshop, crocheting, graphic design) in exchange for pizza.I’m a broke art student/designer with $0.00 in my bank account and no food in the fridge—just finished my last sandwich. If you send me a pizza today, I will  1. Deliver a **high-quality**, **bounded‐scope** piece:      – A custom sketch or digital illustration within 24–48 hours      – A half-finished scarf completed and shipped in 3–4 days (your choice of colors)     – Up to 5 photo retouches or a simple logo/graphic in PSD/AI format   2. Provide **samples or links** up front (Imgur/DeviantArt/Behance) so you know my style.   3. Share an “edit—fulfilled!” update with proof once your pizza arrives.    4. Pay it forward by creating another small piece for someone else when I get paid.Thank you so much—I truly appreciate any help and will make good on every promise.",
    5: "Cluster theme: parents (often single or stay-at-home) facing acute financial hardship who just want to share a pizza night with their young children.I’m completely out of food and money—my last groceries ran out days ago, and with rent, bills, and daycare costs I have nothing left to feed my kids tonight.  If you send us a pizza now, I will: 1. Provide a verification code or pickup details so you know it was used. 2. Share the exact ages and names of my children (e.g., Jason, 6; Michelle & Christina, 3½) to personalize my gratitude. 3. Promise to pay it forward or repay next payday—your kindness will become someone else’s pizza. 4. Publicly thank you by name in an edit when the request is fulfilled.Thank you from the bottom of my heart; helping my family tonight means everything to us.",
    6: "Cluster theme: urgent pet‐related hardship—owners who’ve spent their last dollars on vet bills or pet food, now out of money and food themselves.I’ve just hit $0.00 after paying emergency vet bills and my pantry is completely bare—no ramen, no bread, nothing until payday. My beloved pet (kitten with a broken paw / rescued stray / long‐time companion) needs me, and I’m desperate for a pizza tonight so we both don’t go hungry.If you send a pizza now, I will: 1. Share a photo proof of me and my pet (or vet receipt) immediately upon delivery.   2. Publicly thank you in a thank‐you thread with pet pictures.   3. Pay it forward by covering another RAOP request once I’m back on my feet.Thank you from me and my [cat/kitten/dog]—your kindness literally keeps us alive tonight.",
    7: "Cluster theme: “payday gap” requests—new/shifted job pay schedule leaves you broke until your first or next paycheck.I’ve just started a job (or had my pay delayed) and won’t see any money until [day of week or date]. Right now I’m down to $0–$12 in my bank account, no groceries left, and I need food to get to work/class tomorrow.If you can send me a pizza today, I will: 1. Provide proof on request—bank screenshot, emailed schedule, or verification code—before or after delivery. 2. Clearly state exactly when I’ll pay it forward or repay you (e.g. “I get paid Friday the 16th and will send a gift/card that day”).3. Share a brief edit/update here (“request fulfilled—thank you!”) as social proof. 4. Offer a small token of thanks in return (minor Photoshop work, answering questions, or simply a public shout-out).Thank you so much—I truly appreciate any help and will make good on every promise as soon as my paycheck hits.",
    8: "Cluster theme: deeply personal sob-stories of financial & emotional crisis, pleading for pizza as a vital boost.I have literally $0.00 in my bank and haven’t eaten a real meal in days—I’m surviving on canned beans and rice only. I lost my job/car/home [brief crisis summary], and every bill is past due. Right now I’m at rock bottom and desperately need a pizza to keep me afloat.If you can spare a pie today, I will:1. Send you a verification screenshot or “edit: received!” update immediately upon delivery.  2. Publicly thank you in my next post and commit to paying your kindness forward as soon as I’m back on my feet.  3. Offer a small service (advice, a shout-out, or anything modest) to show my gratitude.Thank you so much—your help today literally means the difference between going hungry or not.  ",
    9: "Cluster theme: highly detailed logistical pizza or e‐gift‐card requests specifying exact brand, price, payment method, and pickup/delivery instructions.I’ve literally run out of food and money until payday—£0.00 in my account and no groceries—and desperately need a pizza today.  If you can help, please note:• Order from Domino’s (personal pizza £5.99) or Papa John’s (one‐topping large £5.99) via dominos.co.uk or the Papa Johns coupon code.  • I can walk 30 min to pick up if paid by card, or provide my PayPal (£3.20 balance) to swap gift-cards at exact value.  • Send me a verification code or PM me for email details to confirm delivery.  • I will absolutely pay your kindness forward the moment I’m back on my feet (gift-card swap, photoshop work, or another pizza).Thank you so much—this precise help will keep me fed until payday and I truly appreciate it.",
}

analyzer = SentimentIntensityAnalyzer()


# data loading
def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["request_time_utc"])
    df["req"] = (
        (df.request_title.fillna("") + " " + df.request_text_edit_aware.fillna(""))
        .str.lower()
        .str.replace(r"[^a-z]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df["target"] = df.requester_received_pizza.astype(int)
    return df


# cleaner loader
def load_and_clean(csv_path: str, text_column: str = "request_text_edit_aware"):
    df = pd.read_csv(csv_path)
    df = df[df[text_column].notna()]
    df = df[df[text_column].str.strip().astype(bool)]

    intro_patterns = [
        r"(?i)^hey(,\s*guys)?:",
        r"(?i)^hi(,\s*everyone)?:",
        r"(?i)^hello(,\s*all)?:",
        r"(?i)\bi love this sub\b",
        r"(?i)\bfirst time (posting|here)\b",
        r"(?i)\bi\.?m in [A-Za-z ]+\b",
    ]
    for pat in intro_patterns:
        df[text_column] = df[text_column].str.replace(pat, "", regex=True)

    df = df.reset_index(drop=True)
    return df, df[text_column].tolist()


#  feature builders
def add_bucket_deciles(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["req_len"] = df2.req.str.split().str.len().replace(0, np.nan)
    for name, pat in BUCKET_PATTERNS.items():
        f = df2.req.str.count(pat) / df2.req_len
        nz = f[f > 0]
        edges = np.quantile(nz, np.linspace(0, 0.9, 10)) if len(nz) else []
        df2[name] = np.where(f == 0, 0, np.digitize(f, edges, right=True) + 1)
    return df2


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["hour"] = d.request_time_utc.dt.hour
    d["hour_sin"] = np.sin(2 * np.pi * d.hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d.hour / 24)
    d["dow"] = d.request_time_utc.dt.weekday
    d["dow_sin"] = np.sin(2 * np.pi * d.dow / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d.dow / 7)
    d["d_comm_age"] = pd.qcut(
        (d.request_time_utc - d.request_time_utc.min()).dt.days, 10, labels=False
    )
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


# SBERT helpers
def embed_texts(texts: list[str], model: str, batch: int = 32) -> np.ndarray:
    mod = SentenceTransformer(model)
    return mod.encode(texts, batch_size=batch, show_progress_bar=True)


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


#  feature pipeline orchestrator
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


def make_feature_matrix(
    df_feat: pd.DataFrame, embs: np.ndarray, fit_pca_on: np.ndarray
) -> tuple[np.ndarray, PCA]:
    bucket_cols = list(BUCKET_PATTERNS.keys())
    sim_cols = [c for c in df_feat.columns if c.startswith("sim_blob_")]
    scalar_cols = [
        "d_comm_age",
        "month_h1",
        "gratitude",
        "hyperlink",
        "reciprocity",
        "sent_pos",
        "sent_neg",
        "req_len",
        "d_karma",
        "posted_before",
        "n_chars",
        "n_words",
        "cnt_please",
        "cnt_thank",
        "vader_sent",
        "vote_sum",
        "vote_diff",
        "vote_ratio",
        "posts_per_day",
        "comments_per_day",
        "votes_per_day",
        "subreddits_per_day",
        "log_account_age",
        "hour_sentiment",
        "length_upvote",
        *[f"sub_{s}" for s in TOP_SUBS],
        "num_top_subs",
        "blob_sim_max",
        "blob_sim_top3_avg",
        "blob_best_id",
    ]
    features = bucket_cols + scalar_cols + sim_cols
    X_tab = df_feat[features].astype(float).values
    pca = PCA(n_components=50, random_state=RND).fit(fit_pca_on)
    X_emb50 = pca.transform(embs)
    return np.hstack([X_tab, X_emb50]), pca, features


#  modelling
def build_stacked_ensemble() -> None:
    print("▶ Loading & preparing data…")
    df_raw, texts = load_and_clean(
        str(DATA_PATH), text_column="request_text_edit_aware"
    )
    df_raw["request_time_utc"] = pd.to_datetime(
        df_raw["request_time_utc"], errors="coerce", utc=True
    )

    df_raw["req"] = [t.lower() for t in texts]
    y = df_raw["requester_received_pizza"].astype(int).values

    # embeddings
    if EMB_CACHE.exists():
        print("▶ Loading cached embeddings")
        embs = np.load(EMB_CACHE)
    else:
        print("▶ Computing embeddings")
        embs = embed_texts(texts, str(EMB_MODEL_PATH))
        np.save(EMB_CACHE, embs)

    idx_train, idx_test = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=RND
    )
    df_feat = build_feature_frame(df_raw, embs)

    Xtrain, pca, feat_names = make_feature_matrix(
        df_feat.iloc[idx_train], embs[idx_train], embs[idx_train]
    )
    Xtest, _, _ = make_feature_matrix(
        df_feat.iloc[idx_test], embs[idx_test], embs[idx_train]
    )

    y_train, y_test = y[idx_train], y[idx_test]

    counts = np.bincount(y_train)
    class_w = {0: len(y_train) / (2 * counts[0]), 1: len(y_train) / (2 * counts[1])}

    pipe_logr = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=2000,
                ),
            ),
        ]
    )
    pipe_gb = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            (
                "gb",
                GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.8,
                    random_state=RND,
                ),
            ),
        ]
    )
    pipe_cat = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            (
                "cat",
                CatBoostClassifier(
                    iterations=500,
                    depth=6,
                    learning_rate=0.01,
                    l2_leaf_reg=3,
                    loss_function="Logloss",
                    eval_metric="AUC",
                    random_seed=RND,
                    class_weights=class_w,
                    logging_level="Silent",
                    allow_writing_files=False,
                ),
            ),
        ]
    )

    stack = StackingClassifier(
        estimators=[
            ("LogReg", pipe_logr),
            ("GradBoost", pipe_gb),
            ("CatBoost", pipe_cat),
        ],
        final_estimator=LogisticRegression(
            penalty="l2", solver="liblinear", class_weight="balanced", max_iter=1000
        ),
        cv=CV,
        n_jobs=-1,
    )

    print("▶ Training on 80 % split …")
    stack.fit(Xtrain, y_train)

    proba = stack.predict_proba(Xtest)[:, 1]
    preds = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    print(f"\n★ Hold-out results on {len(y_test)} samples:")
    print(f"   Accuracy : {acc:0.3f}")
    print(f"   ROC-AUC  : {auc:0.3f}\n")

    print("▶ Re-fitting full model for deployment …")
    Xfull, pca_full, _ = make_feature_matrix(df_feat, embs, embs)
    stack.fit(Xfull, y)

    # save
    with STACK_PATH.open("wb") as f:
        pickle.dump(stack, f)
    meta = {
        "feature_names": feat_names,
        "n_tabular": len(feat_names),
        "pca_components_": pca_full.components_,
        "pca_mean_": pca_full.mean_,
        "embedding_model": str(EMB_MODEL_PATH),
    }
    with META_PATH.open("wb") as f:
        pickle.dump(meta, f)

    print(f"[✓] Model → {STACK_PATH.relative_to(ROOT_DIR)}")
    print(f"[✓] Meta  → {META_PATH.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    build_stacked_ensemble()
