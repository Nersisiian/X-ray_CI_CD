import os
import numpy as np
from PIL import Image


def create_fake_images():
    os.makedirs("data/raw", exist_ok=True)
    # Создаём 3 нормальных и 3 пневмонийных фейковых изображения
    for i in range(3):
        # Нормальное (случайный шум)
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(img).save(f"data/raw/NORMAL_{i}.jpeg")
        # Пневмония (тоже случайный шум, но для разнообразия)
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(img).save(f"data/raw/PNEUMONIA_{i}.jpeg")
    print("Created 3 normal and 3 pneumonia fake images in data/raw/")


if __name__ == "__main__":
    create_fake_images()
