import os
from huggingface_hub import hf_hub_download, list_repo_tree
import shutil

def main():
    repo_id = "angads24/deepfake-video"
    print(f"Connecting to HuggingFace repo: {repo_id}")
    
    real_dir = os.path.join("data", "videos", "real")
    fake_dir = os.path.join("data", "videos", "fake")
    
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # Get lists of files
    fake_files = [f.path for f in list_repo_tree(repo_id, repo_type='dataset', path_in_repo='dataset/fake') if f.path.endswith('.mp4')]
    real_files = [f.path for f in list_repo_tree(repo_id, repo_type='dataset', path_in_repo='dataset/real') if f.path.endswith('.mp4')]
    
    # We download 200 of each to have a solid batch for training without waiting hours
    num_to_download = 200
    
    print(f"Downloading {num_to_download} Fake videos...")
    for fpath in fake_files[:num_to_download]:
        local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=fpath)
        dest_path = os.path.join(fake_dir, os.path.basename(fpath))
        shutil.copy(local_path, dest_path)
        print(f" Downloaded {os.path.basename(fpath)}")
        
    print(f"\nDownloading {num_to_download} Real videos...")
    for fpath in real_files[:num_to_download]:
        local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=fpath)
        dest_path = os.path.join(real_dir, os.path.basename(fpath))
        shutil.copy(local_path, dest_path)
        print(f" Downloaded {os.path.basename(fpath)}")
        
    print("\nDownload complete! Real videos and fake videos are ready in data/videos/")

if __name__ == "__main__":
    main()
