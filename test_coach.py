from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_path = "lucaquillo14/gamesense-football-coach-v2"  # or "./coach_llama3_finetuned/final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")

coach = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "### Instruction:\nHow can a winger improve pressing in a 4-3-3?\n\n### Response:\n"
out = coach(prompt, max_new_tokens=150, temperature=0.7, top_p=0.9)
print(out[0]["generated_text"])