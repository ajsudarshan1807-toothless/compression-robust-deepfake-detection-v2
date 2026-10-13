from huggingface_hub import list_repo_files
files = list_repo_files('OpenRL/DeepFakeFace', repo_type='dataset')
print([f for f in files if f.endswith('.parquet')])
