"""
Phase 2: Preprocessing
"""

import re
import html
import pandas as pd

DATA_DIR = "data/semeval"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"

OUT_TRAIN = f"{DATA_DIR}/train_processed.csv"
OUT_TEST = f"{DATA_DIR}/test_processed.csv"

LABEL_MAP = {"AGAINST": 0, "FAVOR": 1, "NONE": 2}


def normalize(text):
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = re.sub(r"http\S+", "URL", text)
    text = re.sub(r"@\w+", "@USER", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip().lower()

    return text


def build_input(tweet, target):
    return f"{tweet} [SEP] {target}"


def preprocess(df):
    df = df.dropna(subset=["Tweet", "Target", "Stance"]).copy()
    df = df.drop_duplicates(subset=["Tweet", "Target", "Stance"]).copy()

    df["tweet_clean"] = df["Tweet"].apply(normalize)
    df["model_input"] = df.apply(
        lambda r: build_input(r["tweet_clean"], r["Target"]), axis=1
    )
    df["label"] = df["Stance"].map(LABEL_MAP)

    return df


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train = preprocess(train)
    test = preprocess(test)

    train.to_csv(OUT_TRAIN, index=False)
    test.to_csv(OUT_TEST, index=False)

    print("Saved processed files.")


if __name__ == "__main__":
    main()