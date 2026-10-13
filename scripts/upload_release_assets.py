import os
import subprocess
import requests

def get_token():
    cmd = r'''@'
protocol=https
host=github.com
'@ | git credential fill'''
    proc = subprocess.run(["powershell", "-Command", cmd], capture_output=True, text=True, check=True)
    for line in proc.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    return os.environ.get("GITHUB_TOKEN", "")

def main():
    token = get_token()
    if not token:
        print("Error: Could not retrieve GitHub token.")
        return

    release_id = 411564008
    repo = "ajsudarshan1807-toothless/compression-robust-deepfake-detection-v2"
    upload_url_base = f"https://uploads.github.com/repos/{repo}/releases/{release_id}/assets"

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "ReleaseAssetUploader"
    }

    files_to_upload = [
        ("checkpoints/model_i_whatsapp_robust.pt", "model_i_whatsapp_robust.pt"),
        ("checkpoints/video_temporal_model_whatsapp.pt", "video_temporal_model_whatsapp.pt"),
        ("checkpoints/model_a_spatial_clean.pt", "model_a_spatial_clean.pt")
    ]

    for local_path, asset_name in files_to_upload:
        if not os.path.isfile(local_path):
            print(f"Skipping {local_path}: File not found.")
            continue

        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"\nUploading {asset_name} ({size_mb:.2f} MB) to GitHub Release v2.0.0 ...")
        upload_url = f"{upload_url_base}?name={asset_name}"
        
        with open(local_path, "rb") as f:
            upload_headers = headers.copy()
            upload_headers["Content-Type"] = "application/octet-stream"
            resp = requests.post(upload_url, headers=upload_headers, data=f)
            
        if resp.status_code in [200, 201]:
            print(f"[+] Successfully uploaded {asset_name}!")
        else:
            print(f"[!] Upload failed ({resp.status_code}): {resp.text[:200]}")

if __name__ == "__main__":
    main()
