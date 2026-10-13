from datasets import load_dataset
print("Fetching metadata for Hemg dataset...")
try:
    ds = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split='train[:1]')
    item = ds[0]
    print(f"Success! Keys: {item.keys()}")
    print(f"Label: {item.get('label')}")
except Exception as e:
    print(f"Failed: {e}")
