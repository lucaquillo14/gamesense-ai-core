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
DATA_FILE  = os.path.join(os.path.dirname(__file__), "football_trimmed.csv")  # update if different
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
OFFLOAD_DIR = os.path.join(os.path.dirname(__file__), "offload")
os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(OFFLOAD_DIR, exist_ok=True)

print(f"🧠 Loading base model: {BASE_MODEL}")

# ---------------- TOKENIZER -------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
# Ensure a pad token exists for batching
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# ---------------- MODEL (4-bit) ---------
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,              # QLoRA
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    offload_folder=OFFLOAD_DIR,     # spill extra to disk if VRAM tight
)
# Resize embeddings if we added a PAD token
model.resize_token_embeddings(len(tokenizer))

# Save VRAM
model.gradient_checkpointing_enable()
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# ---------------- LoRA ------------------
# For Llama-3, q_proj/v_proj is fine; you can expand to k_proj,o_proj,up/down/gate_proj later.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.train()

print("✅ Model and tokenizer ready.")

# ---------------- DATA ------------------
print(f"📂 Loading dataset: {DATA_FILE}")
ds = load_dataset("csv", data_files={"train": DATA_FILE})

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
