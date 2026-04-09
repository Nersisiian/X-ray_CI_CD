# 🩻 X-Ray Pneumonia Detector — AI-powered Medical Diagnosis

[![CI/CD](https://github.com/Nersisiian/X-ray_CI_CD/actions/workflows/ci.yml/badge.svg)](https://github.com/Nersisiian/X-ray_CI_CD/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow 2.13](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Production-ready web application** for pneumonia detection from chest X‑ray images.  
> Built with TensorFlow, FastAPI, Docker, and full CI/CD pipeline.

## ✨ Key Features

- 🧠 **Deep Learning model** (EfficientNetB0) – classifies X‑ray as NORMAL or PNEUMONIA  
- 🌐 **Web interface** – drag & drop image upload, instant result with confidence score  
- 📡 **REST API** – easy integration with other systems (JSON response)  
- 🐳 **Docker & Docker Compose** – run anywhere with one command  
- 🔁 **Full CI/CD** – automated linting, formatting, tests, model training, Docker build & push  
- 🧪 **Unit & API tests** – ensures reliability and correctness  
- 📊 **Logging & health checks** – production‑ready observability  

## 🚀 Quick Start

### Option 1: Run with Docker (recommended)
`ash
docker-compose up --build
