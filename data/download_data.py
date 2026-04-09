import os
import shutil
import kagglehub


def download_sample():
    print("Downloading Chest X-Ray dataset sample...")
    # Скачиваем полный датасет (около 1GB, но CI возьмёт только пару файлов)
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    print(f"Dataset downloaded to {path}")

    # Копируем несколько примеров в локальную папку data/
    os.makedirs("data/raw", exist_ok=True)
    src = os.path.join(path, "chest_xray/train/NORMAL")
    if os.path.exists(src):
        for f in os.listdir(src)[:3]:  # 3 нормальных
            shutil.copy(os.path.join(src, f), "data/raw/")
    src_pneu = os.path.join(path, "chest_xray/train/PNEUMONIA")
    if os.path.exists(src_pneu):
        for f in os.listdir(src_pneu)[:3]:  # 3 с пневмонией
            shutil.copy(os.path.join(src_pneu, f), "data/raw/")

    print("Sample images saved to data/raw/")


if __name__ == "__main__":
    download_sample()
