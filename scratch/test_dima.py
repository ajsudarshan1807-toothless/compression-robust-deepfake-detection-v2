from datasets import load_dataset
try:
    ds = load_dataset("dima806/deepfake_vs_real_image_detection", split='train', streaming=True)
    item = next(iter(ds))
    print(f"Success! Keys: {item.keys()}")
    print(item)
except Exception as e:
    print(f"Error: {e}")
