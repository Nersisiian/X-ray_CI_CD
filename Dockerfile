FROM tensorflow/tensorflow:2.13.0-gpu AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM tensorflow/tensorflow:2.13.0-gpu
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY . .

RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["python", "-m", "app.main"]