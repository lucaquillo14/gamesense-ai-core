from huggingface_hub import upload_folder, create_repo

# 🧠 Change this only if you want a different name
repo_id = "lucaquillo/gamesense-football-coach-v2"
local_dir = "coach_llama3_finetuned/final"

# ✅ Create repo if it doesn't exist
create_repo(repo_id, repo_type="model", private=True, exist_ok=True)

print(f"🚀 Uploading model from '{local_dir}' to '{repo_id}'...")
upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    repo_type="model"
)
print("✅ Upload complete! Model saved to:", repo_id)
