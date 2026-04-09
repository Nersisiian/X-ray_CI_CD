import argparse
import os
import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import load_and_preprocess, create_augmentation, get_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--sample_fraction", type=float, default=1.0,
                        help="Fraction of dataset to use (for quick CI test)")
    return parser.parse_args()

def load_data(data_dir="data/raw", fraction=1.0):
    images = []
    labels = []
    # предполагаем, что файлы: нормальные - NORMAL*, пневмония - PNEUMONIA*
    for path in glob.glob(os.path.join(data_dir, "*.jpeg")) + glob.glob(os.path.join(data_dir, "*.jpg")):
        if "NORMAL" in path or "normal" in path:
            label = 0
        elif "PNEUMONIA" in path or "pneumonia" in path:
            label = 1
        else:
            continue  # пропустить, если не распознано
        img = load_and_preprocess(path)
        images.append(img)
        labels.append(label)
    images = np.array(images)
    labels = np.array(labels)
    # взять подвыборку
    if fraction < 1.0:
        n = int(len(images) * fraction)
        idx = np.random.choice(len(images), n, replace=False)
        images, labels = images[idx], labels[idx]
    return images, labels

def main():
    args = parse_args()
    print("Loading data...")
    X, y = load_data(fraction=args.sample_fraction)
    if len(X) == 0:
        print("No images found. Run python data/download_data.py first.")
        return

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    aug = create_augmentation()
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y)).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(args.batch_size)

    model = get_model(num_classes=2)
    model.summary()

    print(f"Training for {args.epochs} epochs...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    os.makedirs("models", exist_ok=True)
    model.save("models/xray_model.h5")
    print("Model saved to models/xray_model.h5")

if __name__ == "__main__":
    main()