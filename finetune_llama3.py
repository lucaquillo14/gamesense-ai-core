from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch, os

BASE_MODEL = "lucaquillo/gamesense-football-coach-v2"   # ✅ continue fine-tuning from your v2 model
DATA_FILE  = "football_tactical_coach_3k.jsonl"          # ✅ new dataset
OUTPUT_DIR = "coach_llama3_finetuned_v3"                 # ✅ new output directory

print("🔹 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
tokenizer.padding_side = "right"
pad_id = tokenizer.pad_token_id

# ✅ Load model in 4-bit precision to save VRAM
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# If new tokens were added
model.resize_token_embeddings(len(tokenizer))

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# ✅ LoRA configuration (lightweight adapter fine-tuning)
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)

# ✅ Correct: .train() only applies to the model, not optimizer
model.train()

# Safety & precision tweaks
torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

# ----------------- Dataset -----------------
print(f"📘 Loading dataset from {DATA_FILE}…")
raw = load_dataset("json", data_files={"train": DATA_FILE})["train"]

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
    # Labels same as inputs, ignore padding
    enc["labels"] = []
    for ids in enc["input_ids"]:
        enc["labels"].append([tid if tid != pad_id else -100 for tid in ids])
    return enc

ds = raw.map(tokenize_fn, batched=True, remove_columns=raw.column_names)
print(f"✅ Dataset tokenized: {len(ds)} examples.")

data_collator = default_data_collator

# ----------------- Training -----------------
print("🚀 Starting fine-tuning...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # ✅ higher accumulation, less VRAM
    num_train_epochs=1,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=25,
    save_steps=500,
    save_total_limit=2,
    gradient_checkpointing=True,     # ✅ safe to enable for VRAM savings
    optim="paged_adamw_32bit",       # ✅ correct string, not an object
    report_to="none",
)

import torch.utils.checkpoint
torch.utils.checkpoint.use_reentrant = False  # silence warnings

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=data_collator,
)

trainer.train()

# ----------------- Save -----------------
final_dir = os.path.join(OUTPUT_DIR, "final")
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

print(f"✅ Training complete! Saved to: {final_dir}")
