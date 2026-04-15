import pandas as pd
import os
import time
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import pickle


# =============================================================
# 0. UNIQUE RUN DIRECTORY
# =============================================================
timestamp = time.strftime("%Y%m%d-%H%M%S")
RUN_DIR = f"runs/bert_lora_opt_{timestamp}"

CHECKPOINT_DIR    = f"{RUN_DIR}/checkpoints"
BEST_MODEL_DIR    = f"{RUN_DIR}/best_model"
BASE_MODEL_DIR    = f"{RUN_DIR}/base_model"
LORA_ADAPTER_DIR  = f"{RUN_DIR}/lora_adapter"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(BASE_MODEL_DIR, exist_ok=True)
os.makedirs(LORA_ADAPTER_DIR, exist_ok=True)

print(f"\n📁 RUN DIRECTORY CREATED: {RUN_DIR}\n")


# =============================================================
# 1. LOAD DATA
# =============================================================
train_path = "/home/pnidagun/stance_detection_new/data/semeval/train_processed.csv"
test_path  = "/home/pnidagun/stance_detection_new/data/semeval/test_processed.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

df_train["input"] = df_train["Tweet"] + " [SEP] Target: " + df_train["Target"]
df_test["input"]  = df_test["Tweet"]  + " [SEP] Target: " + df_test["Target"]

label_encoder = LabelEncoder()
df_train["label"] = label_encoder.fit_transform(df_train["Stance"])
df_test["label"]  = label_encoder.transform(df_test["Stance"])

with open(f"{RUN_DIR}/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

train_ds = Dataset.from_pandas(df_train[["input", "label"]])
test_ds  = Dataset.from_pandas(df_test[["input", "label"]])


# =============================================================
# 2. TOKENIZATION
# =============================================================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["input"],
        truncation=True,
        padding="max_length",
        max_length=192,
    )

train_ds = train_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column("label", "labels")
test_ds = test_ds.rename_column("label", "labels")

train_ds.set_format("torch")
test_ds.set_format("torch")


# =============================================================
# 3. LOAD BASE BERT + APPLY LoRA
# =============================================================
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_encoder.classes_)
)

model.save_pretrained(BASE_MODEL_DIR)
tokenizer.save_pretrained(BASE_MODEL_DIR)

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["query", "key", "value"],
    bias="none",
    task_type="SEQ_CLS",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# =============================================================
# 4. CLASS WEIGHTS (normalized)
# =============================================================
class_counts = df_train["label"].value_counts().sort_index()
w = (1 / class_counts)
w = w / w.sum()
weights = torch.tensor(w.values, dtype=torch.float32)

print("\n📊 Class Weights:", weights)


# =============================================================
# 5. CUSTOM TRAINER (FIXES num_items_in_batch issue)
# =============================================================
class WeightedTrainer(Trainer):
    def __init__(self, weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights

    # Accept **kwargs so old transformers do NOT break
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs["logits"]

        loss_fct = torch.nn.CrossEntropyLoss(weight=self.weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# =============================================================
# 6. METRICS
# =============================================================
def compute_metrics(pred):
    logits, labels = pred
    preds = logits.argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


# =============================================================
# 7. TRAINING ARGS — OLD TRANSFORMERS SAFE
# =============================================================
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    learning_rate=1.5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    warmup_ratio=0.06,
    weight_decay=0.01,
    logging_steps=20,
    save_strategy="no",   # we save manually
)


# =============================================================
# 8. TRAINER INIT
# =============================================================
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
    weights=weights,
)


# =============================================================
# 9. MANUAL TRAIN LOOP + BEST MODEL SELECTION
# =============================================================
best_acc = -1
best_epoch = -1
patience = 3
bad_epochs = 0

for epoch in range(training_args.num_train_epochs):
    print(f"\n🔵 Epoch {epoch+1}/{training_args.num_train_epochs}")

    trainer.train()
    metrics = trainer.evaluate()
    print("Eval:", metrics)

    # safe accuracy extraction
    curr_acc = metrics.get("eval_accuracy") or metrics.get("accuracy") or -1
    print(f"📈 Current Accuracy: {curr_acc}")

    # save checkpoint every epoch
    trainer.save_model(f"{CHECKPOINT_DIR}/epoch_{epoch+1}")

    if curr_acc > best_acc:
        best_acc = curr_acc
        best_epoch = epoch + 1
        bad_epochs = 0

        trainer.save_model(BEST_MODEL_DIR)
        tokenizer.save_pretrained(BEST_MODEL_DIR)

        print(f"🏆 New best accuracy: {best_acc:.4f} at epoch {best_epoch}")
    else:
        bad_epochs += 1
        print(f"⚠️ No improvement. Patience left: {patience - bad_epochs}")

    if bad_epochs >= patience:
        print("\n🛑 Early Stopping Activated!")
        break


# =============================================================
# 10. SAVE LORA ADAPTER
# =============================================================
model.save_pretrained(LORA_ADAPTER_DIR)


print("\n🎉 TRAINING COMPLETE!")
print(f"🏆 Best Epoch: {best_epoch}")
print(f"📈 Best Accuracy: {best_acc}")
print(f"📁 Best model saved in: {BEST_MODEL_DIR}")