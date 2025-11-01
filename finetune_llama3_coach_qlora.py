# ------------------------------------------------------------
# Fine-tune LLaMA-3 (8B or smaller) for Football Coaching with QLoRA
# ------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from datasets import load_dataset
import torch, os

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # or TinyLlama/TinyLlama-1.1B-Chat-v1.0
dataset_path = "football_trimmed.csv"  # dataset in same folder

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")

dataset = load_dataset("csv", data_files={"train": dataset_path})

def tokenize_function(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset["train"].map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch")

# Resume from checkpoint if available
output_dir = "./output"
resume_checkpoint = None
if os.path.exists(os.path.join(output_dir, "checkpoint-last")):
    resume_checkpoint = os.path.join(output_dir, "checkpoint-last")

args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=50,
    save_steps=500,
    num_train_epochs=3,
    fp16=True,
    save_total_limit=3,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset
)

trainer.train(resume_from_checkpoint=resume_checkpoint)
trainer.save_model(os.path.join(output_dir, "final_llama3_coach"))

print("✅ Training complete. Model saved to:", output_dir)
