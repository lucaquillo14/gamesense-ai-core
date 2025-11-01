# finetune_llama3_coach_qlora.py
# Fine-tune Meta-Llama-3-8B-Instruct on football coaching data with QLoRA.

from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch, os

# ---------------- CONFIG ----------------
BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"   # chat model for coaching
DATA_FILE  = os.path.join(os.path.dirname(__file__), "football_tactical_technical_5k.jsonl")  # update if different
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "coach_llama3_finetuned")
OFFLOAD_DIR = os.path.join(os.path.dirname(__file__), "offload")
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(OFFLOAD_DIR, exist_ok=True)

print(f"🧠 Loading base model: {BASE_MODEL}")

# ---------------- TOKENIZER -------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# Ensure a pad token exists for batching
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# ---------------- MODEL (4-bit) ---------
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# 🔧 Prepare for 4-bit training
model = prepare_model_for_kbit_training(model)

# 🧩 LoRA config
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)

# ✅ Disable all checkpointing and caching
if hasattr(model, "gradient_checkpointing_disable"):
    model.gradient_checkpointing_disable()
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

model.train()



# ---------------- DATA ------------------
print(f"📂 Loading dataset: {DATA_FILE}")
ds = load_dataset("json", data_files={"train": DATA_FILE})

def tokenize_function(sample):
    text = sample.get("text") or str(sample)
    enc = tokenizer(text, truncation=True, padding="max_length", max_length=256)
    # labels must exist for loss – causal LM learns to predict next token on the same sequence
    enc["labels"] = enc["input_ids"].copy()
    return enc

ds_tok = ds.map(tokenize_function, batched=True, remove_columns=ds["train"].column_names)
print("✅ Dataset tokenized.")

# ---------------- TRAIN -----------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,     # safe on 24GB; raise with A100
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    optim="paged_adamw_32bit",         # memory-friendly
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_tok["train"],
)

print("🚀 Starting fine-tuning...")
trainer.train()
final_dir = os.path.join(OUTPUT_DIR, "final_llama3_coach")
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print("✅ Training complete. Saved to:", final_dir)
