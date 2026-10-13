import os
import urllib.request

def download_video(url, path):
    print(f"Downloading {url} to {path}...")
    try:
        urllib.request.urlretrieve(url, path)
        print("Done.")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def main():
    real_dir = os.path.join("data", "videos", "real")
    fake_dir = os.path.join("data", "videos", "fake")
    
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # We will use two public sample videos as placeholders
    # W3Schools HTML5 Sample Video (A Bear) -> We'll use this as "Real"
    url_real = "https://www.w3schools.com/html/mov_bbb.mp4"
    # Another W3Schools Video (An Earth animation) -> We'll use this as "Fake" placeholder
    url_fake = "https://www.w3schools.com/tags/movie.mp4"
    
    real_path = os.path.join(real_dir, "sample_real.mp4")
    if not os.path.exists(real_path):
        download_video(url_real, real_path)
        
    fake_path = os.path.join(fake_dir, "sample_fake.mp4")
    if not os.path.exists(fake_path):
        download_video(url_fake, fake_path)
        
    print(f"\nCreated dummy video dataset in data/videos/ with {url_real} and {url_fake}.")
    print("These are placeholders to allow the temporal (LSTM) video model to initialize and train.")
    
if __name__ == "__main__":
    main()
