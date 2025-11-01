from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch, os

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DATA_FILE  = "football_tactical_technical_5k.jsonl"
OUTPUT_DIR = "coach_llama3_finetuned"

print("🔹 Loading model and tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
pad_id = tokenizer.pad_token_id

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)

# 🔒 absolutely disable anything checkpoint/caching related
if hasattr(model, "gradient_checkpointing_disable"):
    model.gradient_checkpointing_disable()
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

model.train()

# ----------------- Dataset -----------------
print("📘 Loading dataset…")
raw = load_dataset("json", data_files={"football_tactical_technical_5k.jsonl": DATA_FILE})["train"]

def tokenize_fn(batch):
    texts = [
        f"### Instruction:\n{i}\n\n### Response:\n{r}"
        for i, r in zip(batch["instruction"], batch["response"])
    ]
    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    # labels same as inputs for causal LM
    enc["labels"] = enc["input_ids"].copy()
    return enc

ds = raw.map(
    tokenize_fn,
    batched=True,
    remove_columns=raw.column_names,
)

# Safer collator for left-over padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# ----------------- Training -----------------
print("🚀 Starting fine-tuning…")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=25,
    save_steps=500,
    save_total_limit=2,
    gradient_checkpointing=False,  # ✅ force off
    optim="paged_adamw_32bit",
    report_to="none",
)

# (optional) silence the PyTorch checkpoint warning
import torch.utils.checkpoint
torch.utils.checkpoint.use_reentrant = False

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=data_collator,
)

trainer.train()

# save final
final_dir = os.path.join(OUTPUT_DIR, "final")
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print("✅ Training complete. Saved to:", final_dir)
