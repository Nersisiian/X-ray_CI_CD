# 🩻 X‑ray Pneumonia Detector – Full‑stack ML App with CI/CD

[![CI/CD](https://github.com/YOUR_USERNAME/X-ray_CI_CD/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/X-ray_CI_CD/actions/workflows/ci.yml)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://docker.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green)](https://fastapi.tiangolo.com)

**Production‑ready веб‑приложение** для диагностики пневмонии по рентгеновским снимкам. Включает API, веб‑интерфейс, автоматическое тестирование, Docker‑контейнеризацию и CI/CD пайплайн.

## ✨ Функции
- 📸 Загрузка снимка через веб‑форму или REST API
- 🧠 Классификация: NORMAL / PNEUMONIA с процентом уверенности
- 🐳 Docker Compose – подними сервис одной командой
- 📊 Логирование запросов и ошибок
- 🔁 CI/CD: линтинг, тесты, сборка образа, проверка API
- ⚙️ Конфигурация через переменные окружения

## 🚀 Быстрый старт

### Локально (только API)
```bash
pip install -r requirements.txt
python data/download_data.py
python train.py --epochs 5
python -m app.main
# Откройте http://localhost:8000