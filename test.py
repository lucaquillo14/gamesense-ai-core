from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 🧠 your Hugging Face model repo
MODEL_ID = "lucaquillo/gamesense-football-coach-v2"

print("🔹 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

def ask(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.8,
            top_p=0.9
        )
    print("\n🧠 Coach says:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))

# ⚽ example test
prompt = "### Instruction:\nHow can a winger improve pressing in a 4-3-3?\n\n### Response:\n"
ask(prompt)
