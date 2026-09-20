import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

repo_id = "hvng24/tourism-dataset"
repo_type = "dataset"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repository '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Repository '{repo_id}' not found. Creating it...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, token=os.getenv("HF_TOKEN"))
    print(f"Repository '{repo_id}' created.")

api.upload_folder(
    folder_path="data",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Dataset uploaded to Hugging Face Hub successfully.")
