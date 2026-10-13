from datasets import load_dataset
from huggingface_hub import login

import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    login(token=HF_TOKEN)

try:
    ds = load_dataset("prithivMLmods/Deepfake-vs-Real", split='train', streaming=True, token=HF_TOKEN)
    item = next(iter(ds))
    print(f"Success! Keys: {item.keys()}")
except Exception as e:
    print(f"Error: {e}")
