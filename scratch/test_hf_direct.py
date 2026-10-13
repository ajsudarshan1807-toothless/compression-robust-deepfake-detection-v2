import pandas as pd
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

print("Downloading one parquet file directly via hf_hub_download...")
try:
    file_path = hf_hub_download(repo_id="OpenRL/DeepFakeFace", repo_type="dataset", filename="data/train-00000-of-00021-cc581ec2521fc425.parquet")
    print(f"Downloaded to {file_path}")
    
    # Read the parquet
    table = pq.read_table(file_path)
    df = table.to_pandas()
    print(f"Loaded {len(df)} rows. Columns: {df.columns}")
except Exception as e:
    print(f"Failed: {e}")
