import requests
import io
from PIL import Image

try:
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    r = requests.get("https://thispersondoesnotexist.com/", headers=headers, timeout=10)
    img = Image.open(io.BytesIO(r.content))
    print("Success! Size:", img.size)
except Exception as e:
    print("Failed:", e)
