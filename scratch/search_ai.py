from huggingface_hub import HfApi
api = HfApi()
datasets = api.list_datasets(search="ai generated")
datasets = sorted(list(datasets), key=lambda x: getattr(x, 'downloads', 0), reverse=True)
for d in datasets[:10]:
    print(f"{d.id} - Downloads: {getattr(d, 'downloads', 0)}")
