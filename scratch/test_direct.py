import requests
import json

repo = "Hemg/AI-Generated-vs-Real-Images-Datasets"
print(f"Fetching tree for {repo}...")
url = f"https://huggingface.co/api/datasets/{repo}/tree/main/data"
resp = requests.get(url)
if resp.status_code == 200:
    tree = resp.json()
    for item in tree:
        print(item['path'])
else:
    print(f"Failed: {resp.status_code} {resp.text}")
