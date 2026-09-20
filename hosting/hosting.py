import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

repo_id = "hvng24/visit-with-us"
repo_type = "space"

api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating it...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        space_sdk="docker",
        private=False,
        token=os.getenv("HF_TOKEN"),
    )
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="deployment",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Deployment files pushed to the Hugging Face Space successfully.")
