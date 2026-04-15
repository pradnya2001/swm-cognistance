import torch
import pandas as pd
import numpy as np
import os
import gc
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, PeftModel
from datasets import Dataset
from sklearn.metrics import classification_report
from trl import SFTTrainer

# ==========================================
# 1. CONFIGURATION & AUTHENTICATION
# ==========================================
HF_TOKEN = os.getenv("HF_TOKEN", "")
TRAIN_FILE = "/home/pnidagun/stance_detection_new/data/semeval/train_processed.csv"
TEST_FILE = "/home/pnidagun/stance_detection_new/data/semeval/test_processed.csv"
OUTPUT_DIR = "./llama31_stance_adapters"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
BATCH_SIZE = 2 

SYSTEM_PROMPT = (
    "You are a stance detection system. Determine the stance by reasoning through these 6 steps: "
    "1. Context, 2. Viewpoints, 3. Emotions, 4. Comparison, 5. Logic, 6. Final Stance. "
    "Respond with exactly one label: FAVOR, AGAINST, or NEUTRAL. Use NEUTRAL if no clear stance exists."
)

# ==========================================
# 2. DATA PREPARATION
# ==========================================
def format_llama_chat(examples, tokenizer):
    formatted_texts = []
    for target, tweet, stance in zip(examples['Target'], examples['Tweet'], examples['Stance']):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"TARGET: {target}\nTWEET: {tweet}"},
            {"role": "assistant", "content": f"Reasoning and Stance: {stance}"}
        ]
        formatted_texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
    examples['text'] = formatted_texts
    return examples

def load_and_preprocess_datasets(tokenizer):
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE).replace('NONE', 'NEUTRAL')
    train_df['Stance'] = train_df['Stance'].replace('NONE', 'NEUTRAL').str.upper()
    
    dataset = Dataset.from_pandas(train_df[['Target', 'Tweet', 'Stance']])
    train_dataset = dataset.map(
        lambda x: format_llama_chat(x, tokenizer),
        batched=True,
        remove_columns=['Target', 'Tweet', 'Stance']
    )
    return train_dataset, test_df

# ==========================================
# 3. MODEL SETUP (FIXED RANK & TOKEN)
# ==========================================
def setup_lora_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        token=HF_TOKEN
    )
    
    # FIX: Rank increased to 16 to better capture Neutral class patterns
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    return model, tokenizer, lora_config

# ==========================================
# 4. ROBUST LABEL EXTRACTION
# ==========================================
def extract_label(text):
    text = text.upper()
    if "AGAINST" in text: return "AGAINST"
    if "FAVOR" in text: return "FAVOR"
    if "NEUTRAL" in text or "NONE" in text: return "NEUTRAL"
    return "NEUTRAL" # Default to Neutral if logic is unclear

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    try:
        model, tokenizer, lora_config = setup_lora_model()
        train_ds, test_df = load_and_preprocess_datasets(tokenizer)

        if not os.path.exists(OUTPUT_DIR):
            print("\n--- Starting Fine-Tuning ---")
            trainer = SFTTrainer(
                model=model,
                args=TrainingArguments(
                    output_dir=OUTPUT_DIR, num_train_epochs=10, 
                    per_device_train_batch_size=BATCH_SIZE,
                    fp16=True, save_strategy="epoch", report_to="none"
                ),
                train_dataset=train_ds,
                peft_config=lora_config,
            )
            trainer.train()
            trainer.model.save_pretrained(OUTPUT_DIR)
            
            # Memory Cleanup
            del model, trainer
            gc.collect()
            torch.cuda.empty_cache()
            
            # Reload for inference
            model, tokenizer, _ = setup_lora_model()

        print("\n--- Starting Inference ---")
        model = PeftModel.from_pretrained(model, OUTPUT_DIR).merge_and_unload()
        model.eval()

        test_prompts = test_df.apply(lambda r: tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"TARGET: {r['Target']}\nTWEET: {r['Tweet']}"}
        ], tokenize=False, add_generation_prompt=True), axis=1).tolist()

        predictions = []
        for i in tqdm(range(0, len(test_prompts), 4)):
            batch = test_prompts[i:i+4]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            with torch.no_grad():
                # FIX: Increased to 256 for Chain-of-Stance logic
                outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            
            decoded = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            predictions.extend([extract_label(d) for d in decoded])

        test_df['Predicted'] = predictions
        print("\n===== FINAL CLASSIFICATION REPORT =====")
        print(classification_report(test_df['Stance'].str.upper(), test_df['Predicted'].str.upper(), zero_division=0))

    except Exception as e:
        print(f"Critical Error: {e}")
    finally:
        torch.cuda.empty_cache()
        gc.collect()