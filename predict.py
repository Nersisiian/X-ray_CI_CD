import argparse
import os
import numpy as np
import tensorflow as tf
from utils import load_and_preprocess, IMG_SIZE

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to X-ray image")
    parser.add_argument("--model", default="models/xray_model.h5", help="Path to trained model")
    return parser.parse_args()

def predict(image_path, model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")
    model = tf.keras.models.load_model(model_path)
    img = load_and_preprocess(image_path)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0]
    class_idx = np.argmax(pred)
    classes = ["NORMAL", "PNEUMONIA"]
    confidence = pred[class_idx]
    print(f"Prediction: {classes[class_idx]} (confidence: {confidence:.3f})")
    return classes[class_idx], confidence

if __name__ == "__main__":
    args = parse_args()
    predict(args.image, args.model)