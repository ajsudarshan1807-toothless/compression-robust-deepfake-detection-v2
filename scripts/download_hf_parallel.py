import os
from huggingface_hub import hf_hub_download, list_repo_tree
import shutil
import concurrent.futures

def download_file(args):
    repo_id, fpath, dest_dir = args
    try:
        local_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=fpath)
        dest_path = os.path.join(dest_dir, os.path.basename(fpath))
        shutil.copy(local_path, dest_path)
        return f"Downloaded {os.path.basename(fpath)}"
    except Exception as e:
        return f"Error downloading {os.path.basename(fpath)}: {e}"

def main():
    repo_id = "angads24/deepfake-video"
    print(f"Connecting to HuggingFace repo: {repo_id}")
    
    real_dir = os.path.join("data", "videos", "real")
    fake_dir = os.path.join("data", "videos", "fake")
    
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    print("Fetching file lists...")
    fake_files = [f.path for f in list_repo_tree(repo_id, repo_type='dataset', path_in_repo='dataset/fake') if f.path.endswith('.mp4')]
    real_files = [f.path for f in list_repo_tree(repo_id, repo_type='dataset', path_in_repo='dataset/real') if f.path.endswith('.mp4')]
    
    num_to_download = 200
    fake_tasks = [(repo_id, f, fake_dir) for f in fake_files[:num_to_download]]
    real_tasks = [(repo_id, f, real_dir) for f in real_files[:num_to_download]]
    
    all_tasks = fake_tasks + real_tasks
    
    print(f"Downloading {len(all_tasks)} videos using 20 threads...")
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(download_file, task): task for task in all_tasks}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            completed += 1
            if completed % 10 == 0:
                print(f"Progress: {completed}/{len(all_tasks)} videos downloaded.")

    print("\nDownload complete! Real videos and fake videos are ready in data/videos/")

if __name__ == "__main__":
    main()
