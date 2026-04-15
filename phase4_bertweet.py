import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import pickle
import os
import time


# ======================================================
# 0. CREATE RUN DIRECTORIES
# ======================================================
timestamp = time.strftime("%Y%m%d-%H%M%S")
RUN_DIR = f"runs/bertweet_lora_run_{timestamp}"

CKPT_DIR = f"{RUN_DIR}/checkpoints"
BEST_DIR = f"{RUN_DIR}/best_model"
LORA_DIR = f"{RUN_DIR}/lora_adapter"

os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(BEST_DIR, exist_ok=True)
os.makedirs(LORA_DIR, exist_ok=True)

print(f"\n📁 RUN DIRECTORY CREATED: {RUN_DIR}\n")


# ======================================================
# 1. LOAD DATA
# ======================================================
train_path = "/home/pnidagun/stance_detection_new/data/semeval/train_processed.csv"
test_path  = "/home/pnidagun/stance_detection_new/data/semeval/test_processed.csv"

df_train = pd.read_csv(train_path)
df_test  = pd.read_csv(test_path)

df_train["input"] = df_train["Tweet"] + " [SEP] Target: " + df_train["Target"]
df_test["input"]  = df_test["Tweet"]  + " [SEP] Target: " + df_test["Target"]


# ======================================================
# 2. ENCODE LABELS
# ======================================================
label_encoder = LabelEncoder()
df_train["label"] = label_encoder.fit_transform(df_train["Stance"])
df_test["label"]  = label_encoder.transform(df_test["Stance"])

with open(f"{RUN_DIR}/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

num_labels = len(label_encoder.classes_)


# ======================================================
# 3. DATASETS
# ======================================================
train_ds = Dataset.from_pandas(df_train[["input", "label"]])
test_ds  = Dataset.from_pandas(df_test[["input", "label"]])


# ======================================================
# 4. TOKENIZER
# ======================================================
model_name = "vinai/bertweet-base"

tokenizer = AutoTokenizer.from_pretrained(model_name, normalization=True)

def tokenize(batch):
    return tokenizer(
        batch["input"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column("label", "labels")
test_ds = test_ds.rename_column("label", "labels")

train_ds.set_format("torch")
test_ds.set_format("torch")


# ======================================================
# 5. MODEL + LoRA
# ======================================================
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["query", "key", "value"],
    task_type="SEQ_CLS"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ======================================================
# 6. METRICS FUNCTION
# ======================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    return {
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


# ======================================================
# 7. TRAINING ARGUMENTS (old HF-safe)
# ======================================================
training_args = TrainingArguments(
    output_dir=CKPT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    num_train_epochs=6,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=20,
    save_strategy="no",   # manual saving ONLY
)


# ======================================================
# 8. TRAINER (NO weighted loss required)
# ======================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)


# ======================================================
# 9. MANUAL TRAIN LOOP + BEST WEIGHT SELECTION
# ======================================================
best_acc = -1
best_epoch = -1

for epoch in range(training_args.num_train_epochs):
    print(f"\n🔵 Epoch {epoch+1}/{training_args.num_train_epochs}")

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval:", metrics)

    curr_acc = metrics.get("eval_accuracy") or metrics.get("accuracy") or -1
    print(f"📈 Current Accuracy: {curr_acc}")

    # save epoch weights
    trainer.save_model(f"{CKPT_DIR}/epoch_{epoch+1}")

    # best model logic
    if curr_acc > best_acc:
        best_acc = curr_acc
        best_epoch = epoch + 1

        trainer.save_model(BEST_DIR)
        tokenizer.save_pretrained(BEST_DIR)

        print(f"🏆 New BEST model at epoch {best_epoch} — acc: {best_acc:.4f}")


# ======================================================
# 10. SAVE LoRA adapter separately
# ======================================================
model.save_pretrained(LORA_DIR)


# ======================================================
# 11. FINAL METRICS
# ======================================================
metrics = trainer.evaluate()
print("\n📌 FINAL METRICS:", metrics)

pred_output = trainer.predict(test_ds)
preds = np.argmax(pred_output.predictions, axis=1)
labels = pred_output.label_ids

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(labels, preds, target_names=label_encoder.classes_))

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(labels, preds))


print("\n🎉 Training complete — BEST model saved successfully!")
print(f"🏆 BEST ACCURACY = {best_acc}")
print(f"📁 Best model in: {BEST_DIR}")