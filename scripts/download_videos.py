import os
from huggingface_hub import list_repo_tree

def check_repo(repo_id):
    print(f"\nChecking repo: {repo_id}")
    try:
        files = list_repo_tree(repo_id, repo_type="dataset")
        video_files = [f.path for f in files if f.path.endswith(('.mp4', '.avi', '.zip'))]
        print(f"Found {len(video_files)} files: {video_files[:10]}")
    except Exception as e:
        print(f"Failed to list repo {repo_id}: {e}")

if __name__ == "__main__":
    repos = [
        'hruthik07/Deep-Fake-video', 
        'bigdong89/Deepfake-Videos-2025', 
        'liusiyi641/deep_fake_videos', 
        'Sarim-Hash/video_DEEPFAKE_dataset',
        'belkhir-nacim/deepfake-videos'
    ]
    for r in repos:
        check_repo(r)
