from datasets import load_dataset
for ds_name in ["OpenRL/DeepFakeFace", "tanganke/deepfake_faces", "danjacobellis/deepfake_faces", "saakshigupta/deepfake-detection-dataset-v3"]:
    try:
        print(f"Testing {ds_name}")
        ds = load_dataset(ds_name, split='train', streaming=True)
        row = next(iter(ds))
        print("Keys:", row.keys())
        break
    except Exception as e:
        print("Failed:", e)
