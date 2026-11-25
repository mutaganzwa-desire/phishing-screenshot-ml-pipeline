"""Flood the prediction API with screenshot requests.

Headless example (what locust/run_scaling_test.sh does):
  locust -f locust/locustfile.py --host http://localhost:8080 \
         --headless -u 50 -r 10 -t 60s --csv locust/results/4_containers
"""
import random
from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parents[1]
IMAGES = sorted((ROOT / "data" / "test").rglob("*.jpg"))[:200]
PAYLOADS = [(p.name, p.read_bytes()) for p in IMAGES]


class ScreenshotUser(HttpUser):
    wait_time = between(0.5, 1.5)

    @task(10)
    def predict(self):
        name, data = random.choice(PAYLOADS)
        with self.client.post("/predict", files={"file": (name, data, "image/jpeg")},
                              name="/predict", catch_response=True) as r:
            if r.status_code != 200 or "label" not in r.json():
                r.failure(f"bad response {r.status_code}")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
