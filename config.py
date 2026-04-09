import os
from pathlib import Path
from dotenv import load_dotenv

# Загружаем переменные из .env файла (если есть)
load_dotenv()

# Базовая директория проекта (где находится config.py)
BASE_DIR = Path(__file__).resolve().parent

# Путь к сохранённой модели
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "xray_model.h5"))

# Порт для API
API_PORT = int(os.getenv("API_PORT", "8000"))

# Уровень логирования (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()

# Redis URL (опционально, для кеширования)
REDIS_URL = os.getenv("REDIS_URL", None)
