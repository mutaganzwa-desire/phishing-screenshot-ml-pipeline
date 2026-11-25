FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8000 WORKERS=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache the ImageNet weights so retraining never needs the internet at runtime.
RUN python -c "import keras; keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224,224,3))"

COPY src ./src
COPY models ./models
COPY data ./data
COPY reports ./reports

# Run as a non-root user (uid 1000) that
# can write new uploads and model versions.
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/health')"
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS}"]
