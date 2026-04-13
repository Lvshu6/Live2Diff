from huggingface_hub import HfApi
api = HfApi()

# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
api.upload_large_folder(
    folder_path="/home/suat/yxd/DiffSynth-Studionew/WanFlow",
    repo_id="lvshu0007/WanFlow",
    repo_type="model",
)