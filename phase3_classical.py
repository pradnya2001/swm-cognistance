"""
Phase 3: Classical ML
LR + Naive Bayes + Random Forest
"""

import os
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

DATA_DIR = "data/semeval"
TRAIN_PATH = f"{DATA_DIR}/train_processed.csv"
TEST_PATH = f"{DATA_DIR}/test_processed.csv"

os.makedirs("models/classical", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("results", exist_ok=True)

LABELS = ["AGAINST", "FAVOR", "NONE"]
LABEL_IDS = [0, 1, 2]


def load():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test


def vectorize(train_text, test_text):
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train = tfidf.fit_transform(train_text)
    X_test = tfidf.transform(test_text)
    joblib.dump(tfidf, "models/classical/tfidf.pkl")
    return X_train, X_test


def evaluate(y_true, y_pred, name, save_prefix):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_IDS, zero_division=0
    )

    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=LABEL_IDS)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=LABELS,
        yticklabels=LABELS,
        cmap="Blues",
        ax=ax
    )
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(f"plots/{save_prefix}_cm.png", dpi=150)
    plt.close()

    return {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Macro-F1": round(macro_f1, 4),
        "Weighted-F1": round(weighted_f1, 4),
        "AGAINST-F1": round(f[0], 4),
        "FAVOR-F1": round(f[1], 4),
        "NONE-F1": round(f[2], 4),
    }


def main():
    train, test = load()

    X_train, X_test = vectorize(train["model_input"].fillna(""), test["model_input"].fillna(""))
    y_train = train["label"].values
    y_test = test["label"].values

    results = []

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    joblib.dump(lr, "models/classical/logistic_regression.pkl")
    results.append(evaluate(y_test, lr.predict(X_test), "Logistic Regression", "logistic_regression"))

    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train, y_train)
    joblib.dump(nb, "models/classical/naive_bayes.pkl")
    results.append(evaluate(y_test, nb.predict(X_test), "Naive Bayes", "naive_bayes"))

    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, "models/classical/random_forest.pkl")
    results.append(evaluate(y_test, rf.predict(X_test), "Random Forest", "random_forest"))

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/classical_results.csv", index=False)
    print("\nSaved results to results/classical_results.csv")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()