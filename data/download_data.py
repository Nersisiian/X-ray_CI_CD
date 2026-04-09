import os
import shutil
import kagglehub

def download_sample():
    print("Downloading Chest X-Ray dataset sample...")
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print(f"Dataset downloaded to {path}")

    os.makedirs("data/raw", exist_ok=True)
    src_normal = os.path.join(path, "chest_xray/train/NORMAL")
    src_pneu = os.path.join(path, "chest_xray/train/PNEUMONIA")

    if os.path.exists(src_normal):
        for f in os.listdir(src_normal)[:3]:
            shutil.copy(os.path.join(src_normal, f), "data/raw/")
    if os.path.exists(src_pneu):
        for f in os.listdir(src_pneu)[:3]:
            shutil.copy(os.path.join(src_pneu, f), "data/raw/")

    print("Sample images saved to data/raw/")

if __name__ == "__main__":
    download_sample()
