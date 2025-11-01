# finetune_llama3_coach_qlora.py
# -----------------------------------------
# Fine-tunes an open LLM (Mistral or TinyLlama) on football coaching data

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import os

# ---------------- CONFIG -----------------
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"   # fallback: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_FILE = "football_small.csv"                    # your dataset file
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD MODEL -------------
print(f"🧠 Loading base model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    offload_folder="./offload",
)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# ---------------- PEFT / LoRA -------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# ---------------- DATA --------------------
dataset = load_dataset("csv", data_files={"train": DATA_FILE})
def format_sample(sample):
    text = sample["text"] if "text" in sample else str(sample)
    return {"input_ids": tokenizer(text, truncation=True, padding="max_length", max_length=256)["input_ids"]}
dataset = dataset.map(format_sample, batched=False)

# ---------------- TRAINING ----------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    optim="paged_adamw_32bit",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

print("🚀 Starting fine-tuning...")
trainer.train()
model.save_pretrained(os.path.join(OUTPUT_DIR, "final_mistral_coach"))
print("✅ Training complete. Model saved to:", os.path.join(OUTPUT_DIR, "final_mistral_coach"))
