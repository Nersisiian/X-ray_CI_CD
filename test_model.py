import os
import pytest
import numpy as np
import tensorflow as tf
from utils import load_and_preprocess, get_model


def test_preprocessing():
    # создадим фейковое изображение
    fake_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    path = "test_fake.jpg"
    tf.keras.preprocessing.image.save_img(path, fake_img)
    processed = load_and_preprocess(path)
    os.remove(path)
    assert processed.shape == (224, 224, 3)
    assert np.max(processed) <= 1.0


def test_model_creation():
    model = get_model(num_classes=2)
    assert model.output_shape == (None, 2)
    assert len(model.layers) > 0


def test_model_save_load(tmp_path):
    model = get_model(num_classes=2)
    save_path = tmp_path / "test_model.h5"
    model.save(save_path)
    loaded = tf.keras.models.load_model(save_path)
    assert loaded.input_shape == model.input_shape


def test_prediction_pipeline():
    # минимальный энд-ту-энд тест
    from train import load_data, get_model

    X, y = load_data(fraction=0.1)
    if len(X) == 0:
        pytest.skip("No test images found, run download_data.py first")
    model = get_model(num_classes=2)
    # прогоним один батч без обучения
    out = model.predict(np.expand_dims(X[0], axis=0))
    assert out.shape == (1, 2)
