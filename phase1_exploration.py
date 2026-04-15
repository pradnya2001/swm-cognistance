"""
Phase 1: Data Exploration (SemEval only)
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data/semeval"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"

random.seed(42)
np.random.seed(42)

os.makedirs("plots", exist_ok=True)


def load_data():
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    return train, test


def basic_stats(df, name):
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nNulls:\n{df.isnull().sum()}")
    print(f"\nDuplicates: {df.duplicated().sum()}")


def class_distribution(df, name, prefix):
    dist = df["Stance"].value_counts().reindex(["AGAINST", "FAVOR", "NONE"], fill_value=0)
    print(f"\n{name} Distribution:\n{dist}")

    fig, ax = plt.subplots()
    dist.plot(kind="bar", ax=ax)
    ax.set_title(name)
    plt.tight_layout()
    plt.savefig(f"plots/{prefix}_dist.png")
    plt.close()


def explore():
    train, test = load_data()

    basic_stats(train, "Train")
    basic_stats(test, "Test")

    print("\nTargets:", train["Target"].unique())

    class_distribution(train, "Train", "train")
    class_distribution(test, "Test", "test")


if __name__ == "__main__":
    explore()