import hashlib
from pathlib import Path

import pandas as pd

RAW = Path(
    r"C:\Users\mzouicha\OneDrive - Amadeus Workplace\Desktop\STAGE\raop-pizza\data\raw\dataset.json"
)  # ajuste si besoin
OUT = Path(
    r"C:\Users\mzouicha\OneDrive - Amadeus Workplace\Desktop\STAGE\raop-pizza\data\processed\dataset_clean.csv"
)
TS_COL = "unix_timestamp_of_request_utc"


def load_raw(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_json(path, orient="records")
        print(f"DataFrame columns: {df.columns.tolist()}")
        print(f"DataFrame shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading JSON: {e}")
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return pd.DataFrame(data)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["giver_username_if_known"].replace("N/A", pd.NA, inplace=True)

    df["post_was_edited"] = df["post_was_edited"].astype(bool)

    df["request_time_utc"] = pd.to_datetime(df[TS_COL], unit="s", utc=True)
    df.drop(columns=[TS_COL, "unix_timestamp_of_request"], inplace=True)

    text_columns = ["request_text", "request_text_edit_aware", "request_title"]
    for col in text_columns:
        if col in df.columns:
            # Remove None values
            df[col] = df[col].fillna("")

            df[f"{col}_original"] = df[col].copy()

            # Convert to lowercase
            df[col] = df[col].str.lower()

            # Remove URLs
            df[col] = df[col].str.replace(r"http\S+|www\S+", "", regex=True)

            # Remove extra whitespace
            df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()

    for col in text_columns:
        if col in df.columns:
            df[f"{col}_length"] = df[col].str.len()
            df[f"{col}_word_count"] = df[col].str.split().str.len()

    return df


def save(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return hashlib.md5(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    print("Lecture…")
    raw = load_raw(RAW)
    print(f"{raw.shape[0]} lignes")

    print("Nettoyage minimal…")
    clean_df = clean(raw)

    print("Sauvegarde…")
    md5 = save(clean_df, OUT)
    print(f"Csv enregistré → {OUT}  (md5={md5})")
