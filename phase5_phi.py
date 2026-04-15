import os
import gc
import time
import torch
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# ==========================================
# 1. CONFIGURATION & FILE PATHS
# ==========================================
TRAIN_FILE = "/home/pnidagun/stance_detection_new/data/semeval/train_processed.csv"
TEST_FILE = "/home/pnidagun/stance_detection_new/data/semeval/test_processed.csv"
OUTPUT_DIR = "./phi3_stance_adapters"
PREDICTION_FILE = "test_predictions.csv"
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
BATCH_SIZE = 4 

# ==========================================
# 2. DATA PREPARATION (COHERENT WITH CoS)
# ==========================================

def prepare_data(df):
    """
    Formats the DataFrame rows for CoS fine-tuning by embedding the full 6-step logic.
    """
    df['Stance'] = df['Stance'].replace('NONE', 'NEUTRAL').str.upper()

    df['text'] = df.apply(
        lambda row: (
            f"<|system|>You are a stance detection system trained on the Chain-of-Stance (CoS) methodology. Your goal is to determine the stance of the TWEET toward a given TARGET by executing the following reasoning pipeline internally:\n"
            f"Step 1: Understand the contextual information (identity, audience, socio-cultural background).\n"
            f"Step 2: Interpret the main ideas and core viewpoints in the TWEET (V).\n"
            f"Step 3: Analyze the language expression and emotional attitude (E).\n"
            f"Step 4: Compare the TWEET with all three possible stances (Favor, Against, Neutral) based on V and E.\n"
            f"Step 5: Conduct logical inference to confirm the consistency and rationality of the stance.\n"
            f"Step 6: Based on this entire process, determine the Final Stance polarity.\n"
            f"Respond ONLY with the Final Stance label (FAVOR, AGAINST, or NEUTRAL).<|end|>"
            f"<|user|>TARGET: {row['Target']}\nTWEET: {row['Tweet']}<|end|>"
            f"<|assistant|>{row['Stance']}<|end|>"
        ), axis=1
    )
    return Dataset.from_pandas(df[['text']])

def load_and_preprocess_datasets():
    if not all([os.path.exists(TRAIN_FILE), os.path.exists(TEST_FILE)]):
        raise FileNotFoundError(f"Ensure {TRAIN_FILE} and {TEST_FILE} are available in the current directory.")

    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    test_df['Stance'] = test_df['Stance'].replace('NONE', 'NEUTRAL').str.upper()

    if os.path.exists(OUTPUT_DIR):
        print(f"Adapters found at {OUTPUT_DIR}. Skipping fine-tuning.")
        return None, test_df
    else:
        print("Preparing training data with full CoS instruction set...")
        train_dataset = prepare_data(train_df)
        return train_dataset, test_df

# ==========================================
# 3. MODEL AND LORA SETUP
# ==========================================

def setup_lora_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return model, tokenizer, lora_config

# ==========================================
# 4. TRAINING
# ==========================================

def fine_tune(model, tokenizer, lora_config, train_dataset):
    print("\n--- Starting LoRA Fine-Tuning ---")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        fp16=True,
        optim="adamw_torch",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        report_to="none"
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        peft_config=lora_config,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ Fine-Tuning complete. Adapters saved to {OUTPUT_DIR}")

# ==========================================
# 5. INFERENCE & EVALUATION
# ==========================================

def run_batch_inference(model, tokenizer, test_df):
    print("\n--- Starting Batch Inference ---")
    test_prompts = test_df.apply(
        lambda row: (
            f"<|system|>You are a stance detection system trained on the Chain-of-Stance (CoS) methodology. Your goal is to determine the stance of the TWEET toward a given TARGET by executing the following reasoning pipeline internally:\n"
            f"Step 1: Understand the contextual information (identity, audience, socio-cultural background).\n"
            f"Step 2: Interpret the main ideas and core viewpoints in the TWEET (V).\n"
            f"Step 3: Analyze the language expression and emotional attitude (E).\n"
            f"Step 4: Compare the TWEET with all three possible stances (Favor, Against, Neutral) based on V and E.\n"
            f"Step 5: Conduct logical inference to confirm the consistency and rationality of the stance.\n"
            f"Step 6: Based on this entire process, determine the Final Stance polarity.\n"
            f"Respond ONLY with the Final Stance label (FAVOR, AGAINST, or NEUTRAL).<|end|>"
            f"<|user|>TARGET: {row['Target']}\nTWEET: {row['Tweet']}<|end|><|assistant|>"
        ), axis=1
    ).tolist()

    predictions = []
    for i in tqdm(range(0, len(test_prompts), 8), desc="Batch Testing"):
        batch = test_prompts[i : i + 8]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        
        decoded = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        predictions.extend([d.strip() for d in decoded])
    return predictions

def evaluate_and_report(test_df, predictions_raw):
    def extract_label(text):
        try:
            stance = text.split()[0].strip().upper().replace('.', '')
            return stance if stance in ['FAVOR', 'AGAINST', 'NEUTRAL'] else "ERROR"
        except: return "ERROR"

    y_pred = pd.Series([extract_label(p) for p in predictions_raw]).str.lower()
    y_true = test_df['Stance'].str.lower()
    
    labels = ['favor', 'against', 'neutral']
    mask = y_pred.isin(labels)
    
    if mask.any():
        print("\n===== CLASSIFICATION REPORT =====")
        print(classification_report(y_true[mask], y_pred[mask], labels=labels, zero_division=0))
        test_df['Predicted_Stance'] = y_pred
        test_df.to_csv(PREDICTION_FILE, index=False)
    else:
        print("No valid predictions found.")

# ==========================================
# 6. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    try:
        train_dataset, test_df = load_and_preprocess_datasets()
        model, tokenizer, lora_config = setup_lora_model()

        if train_dataset:
            fine_tune(model, tokenizer, lora_config, train_dataset)

        # Merge for inference
        try:
            model = PeftModel.from_pretrained(model, OUTPUT_DIR)
            model = model.merge_and_unload()
            print("✅ Weights merged.")
        except Exception as e:
            print(f"Inference error: {e}")

        raw_preds = run_batch_inference(model, tokenizer, test_df)
        evaluate_and_report(test_df, raw_preds)

        # --- Colab Specific Download Block ---
        ZIP_NAME = "phi3_adapters_backup.zip"
        if os.path.exists(OUTPUT_DIR):
            print(f"\nZipping {OUTPUT_DIR}...")
            shutil.make_archive(ZIP_NAME.replace('.zip',''), 'zip', OUTPUT_DIR)
            
            try:
                from google.colab import files
                print("Colab environment detected. Initiating download...")
                files.download(ZIP_NAME)
            except ImportError:
                print(f"Non-Colab environment. Archive saved as: {os.path.abspath(ZIP_NAME)}")

    except Exception as e:
        print(f"Critical Failure: {e}")