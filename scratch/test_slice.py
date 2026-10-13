from datasets import load_dataset
print("Fetching 500 images without streaming...")
ds = load_dataset('OpenRL/DeepFakeFace', split='train[:500]')
print(f"Downloaded! Has {len(ds)} images")
