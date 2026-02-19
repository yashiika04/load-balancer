import time
import random
import sys

from flask import Flask, jsonify, Response, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from prometheus_client import Counter, Histogram
from prometheus_flask_exporter import PrometheusMetrics


app = Flask(__name__)

# Server port (default)
SERVER_PORT = 8000

# Rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["5 per second"],
)

# Prometheus auto metrics
prom_metrics = PrometheusMetrics(app)

# Histogram for Request Response Time (in seconds)
req_response_time = Histogram(
    "http_flask_req_resp_time_seconds",
    "Time taken in request-response by different routes.",
    ["method", "route", "status_code"],
    buckets=[0.1, 0.5, 1, 2, 3, 5]
)

# Counter for Total Requests
total_req_counter = Counter(
    "total_requests",
    "Total requests made to the server."
)


# ------------------ Middleware ------------------

@app.before_request
def start_timer():
    request.start_time = time.time()  # type: ignore


@app.after_request
def log_request(response):
    total_req_counter.inc()

    duration = time.time() - getattr(request, "start_time", time.time())

    req_response_time.labels(
        method=request.method,
        route=request.path,
        status_code=str(response.status_code),
    ).observe(duration)

    return response


# ------------------ Heavy Operation ------------------

def heavy_operation():
    """Simulate a heavy computation with a random delay"""
    delay = random.uniform(0.5, 2)
    time.sleep(delay)
    return "OK!!"


# ------------------ Routes ------------------

@app.route("/")
def index():
    return jsonify({"message": f"Server running on port {SERVER_PORT}"})


# Server capacity limits
server_capacity = {
    8000: 15,
    8001: 10,
    8002: 5,
}


@app.route("/heavy-task")
@limiter.limit(lambda: f"{server_capacity.get(SERVER_PORT, 5)} per second")
def heavy_task():
    try:
        return heavy_operation()
    except Exception as e:
        return str(e), 500


@app.route("/metrics")
def metrics_endpoint():
    try:
        metrics_data = generate_latest(REGISTRY)
        return Response(metrics_data, mimetype=CONTENT_TYPE_LATEST)
    except Exception as e:
        return str(e), 500


# ------------------ Run ------------------

if __name__ == "__main__":
    SERVER_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    app.run(host="0.0.0.0", port=SERVER_PORT)
