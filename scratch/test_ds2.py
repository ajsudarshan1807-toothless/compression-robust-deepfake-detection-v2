from datasets import load_dataset
try:
    ds = load_dataset("OpenRL/DeepFakeFace", split='train', streaming=True)
    row = next(iter(ds))
    print(row.keys())
    # print type of image
    print(type(row.get('image', None)))
except Exception as e:
    print("Error:", e)
