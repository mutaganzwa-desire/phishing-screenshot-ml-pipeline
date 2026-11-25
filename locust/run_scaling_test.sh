#!/usr/bin/env bash
# Run the same Locust load against 1, 2 and 4 API containers and save results.
# Usage: bash locust/run_scaling_test.sh [users] [duration]
set -euo pipefail
cd "$(dirname "$0")/.."
USERS=${1:-50}
DURATION=${2:-60s}
mkdir -p locust/results
docker compose up -d --build --scale api=1
for N in 1 2 4; do
  echo "=== $N container(s) ==="
  docker compose up -d --scale api=$N --no-recreate
  # wait until every replica reports healthy
  until [ "$(docker compose ps api --format '{{.Health}}' | grep -c healthy)" -ge "$N" ]; do sleep 3; done
  sleep 6   # let nginx re-resolve the new replicas
  locust -f locust/locustfile.py --host http://localhost:8080 --headless \
    -u "$USERS" -r 10 -t "$DURATION" --csv "locust/results/${N}_containers" --only-summary
done
python locust/summarize.py
