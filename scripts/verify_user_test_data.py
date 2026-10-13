import os
import sys
from pathlib import Path
from PIL import Image
from deepfake_detector.infer.predictor import Predictor

def main():
    base = Path(r"C:\Users\sudarshan\OneDrive\Pictures\ai testing data")
    print(f"Initializing Predictor for test folder: {base}")
    predictor = Predictor(device_name="cpu")
    print(f"Predictor loaded successfully: Model Version = {predictor.model_version}")

    categories = [
        ("real image", "REAL", False),
        ("fake AI created image", "FAKE", False),
        ("real video", "REAL", True),
        ("fake AI created video", "FAKE", True)
    ]

    results = []
    for folder, ground_truth, is_video in categories:
        fdir = base / folder
        if not fdir.is_dir():
            print(f"Warning: directory not found: {fdir}")
            continue

        files = sorted([f for f in fdir.iterdir() if not f.name.startswith('.')])
        print(f"\n=======================================================")
        print(f"Testing {folder} ({len(files)} files) | Expected: {ground_truth}")
        print(f"=======================================================")

        for fpath in files:
            try:
                if is_video:
                    res = predictor.analyze_video(fpath, num_frames=8)
                    p = res.p_fake_calibrated
                    verdict = res.state
                    frames_info = f"frames={res.sampled_frame_count}"
                else:
                    img = Image.open(fpath).convert("RGB")
                    res = predictor.analyze_image(img, generate_heatmap=False, generate_spectrum=False)
                    p = res.p_fake_calibrated
                    verdict = res.state
                    face_info = "face=YES" if res.face_box else "face=NO"
                    frames_info = face_info

                correct = (verdict == ground_truth)
                results.append({
                    "folder": folder,
                    "file": fpath.name,
                    "truth": ground_truth,
                    "predicted": verdict,
                    "prob": p,
                    "correct": correct
                })
                status = "PASS" if correct else "FAIL"
                print(f"  [{status}] {fpath.name:<45} -> {verdict:<5} (p_fake={p:.4f}, {frames_info})")
            except Exception as e:
                print(f"  [ERROR] {fpath.name}: {e}")
                results.append({
                    "folder": folder,
                    "file": fpath.name,
                    "truth": ground_truth,
                    "predicted": f"ERROR: {e}",
                    "prob": -1.0,
                    "correct": False
                })

    total = len(results)
    correct_cnt = sum(1 for r in results if r["correct"])
    accuracy = (correct_cnt / total * 100) if total > 0 else 0

    print("\n" + "="*70)
    print("FINAL ACCURACY VERIFICATION REPORT")
    print("="*70)
    for folder, ground_truth, is_video in categories:
        cat_res = [r for r in results if r["folder"] == folder]
        cat_correct = sum(1 for r in cat_res if r["correct"])
        cat_tot = len(cat_res)
        cat_acc = (cat_correct / cat_tot * 100) if cat_tot > 0 else 0
        print(f"{folder:<26}: {cat_correct}/{cat_tot} ({cat_acc:6.2f}%)")

    print("-" * 70)
    print(f"OVERALL EVALUATION ACCURACY: {correct_cnt}/{total} ({accuracy:6.2f}%)")
    print("="*70)

if __name__ == "__main__":
    main()
