from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "lucaquillo/gamesense-football-coach-v2"

print("🔹 Loading model (4-bit)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

def ask(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.8,
            top_p=0.9
        )
    print("\n🧠 Coach says:\n", tokenizer.decode(output[0], skip_special_tokens=True))

ask("### Instruction:\nHow can a winger improve pressing in a 4-3-3?\n\n### Response:\n")
