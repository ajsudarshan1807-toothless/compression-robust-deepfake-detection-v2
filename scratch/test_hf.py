from datasets import load_dataset
import traceback

def test_ds(name, *args, **kwargs):
    print(f"Testing {name}...")
    try:
        ds = load_dataset(name, *args, split='train', streaming=True, **kwargs)
        item = next(iter(ds))
        print(f"Success! Keys: {item.keys()}")
        if 'label' in item:
            print(f"Label sample: {item['label']}")
        return True
    except Exception as e:
        print(f"Failed: {e}")
        return False

# Test Cybersentinal
test_ds('vjjoshi23/Deepfake_dataset_cybersentinal', 'core')

# Test alternative deepfake sets
test_ds('DeepFake-Detection/Celeb-DF-v2')
test_ds('Hemgg/deep-fake-detection-dfd-entire-original-dataset')
