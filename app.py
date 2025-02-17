import time
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, jsonify, Response, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import requests
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry, REGISTRY
  
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_flask_exporter import PrometheusMetrics

limiter = Limiter(key_func=get_remote_address)

app = Flask(__name__)
limiter.init_app(app)   

metrics = PrometheusMetrics(app)
 
# Histogram for Request Response Time
req_response_time = Histogram(
    "http_flask_req_resp_time",
    "Total time taken in request-response by different routes.",
    ["method", "route", "statusCode"],
    buckets=[1, 5, 10, 15, 20, 40, 80, 100, 200, 500]
)

# Counter for Total Requests
total_req_counter = Counter(
    "total_requests",
    "Total requests made to the server."
)
 
# Middleware to Track Request Time
@app.before_request
def start_timer():
    request.start_time = time.time()


@app.after_request
def log_request(response):
    total_req_counter.inc()  # Increment request counter
    duration = time.time() - request.start_time  # Calculate time taken

    req_response_time.labels(
        method=request.method,
        route=request.path,
        statusCode=response.status_code
    ).observe(duration)

    return response


# --- Heavy Operation ---

def heavyOperation(): 
    # Random delay between 0 and 4 seconds
    delay = random.random() * 4
    time.sleep(delay)
    return "OK!!"

# --- Endpoints ---

@app.route("/")
def index():
    return jsonify({"message": "Ok!"})

# The /heavy-task endpoint uses the rate limiter.
@app.route("/heavy-task")
@limiter.limit("10 per second")
def heavy_task():
    try:
        result = heavyOperation()
        return result
    except Exception as e:
        return str(e), 500

# The /simulate-traffic endpoint fires 15 parallel requests to /heavy-task.
@app.route("/simulate-traffic")
def simulate_traffic():
    MAX_REQUESTS = 15
    responses = []
    base_url = "http://localhost:8000/heavy-task"  

    # Using a ThreadPoolExecutor to make parallel HTTP requests.
    with ThreadPoolExecutor(max_workers=MAX_REQUESTS) as executor:
        # Dictionary to track which future corresponds to which API call id.
        futures = {executor.submit(requests.get, base_url): i for i in range(MAX_REQUESTS)}
        for future in as_completed(futures):
            call_id = futures[future]
            try:
                response = future.result()
                if response.status_code == 200:
                    responses.append(f"API Call Id {call_id} Response: {response.text}\n")
                else:
                    responses.append(f"API Call Id {call_id} Failed: Request failed with status code {response.status_code}\n")
            except Exception as e:
                responses.append(f"API Call Id {call_id} Failed: {str(e)}\n")
 
    return "\n".join(responses)

# The /metrics endpoint exposes Prometheus metrics.
@app.route("/metrics")
def metrics():
    try:
        # generate_latest() fetches all the default metrics.
        metrics_data = generate_latest(REGISTRY)    
        return Response(metrics_data, mimetype=CONTENT_TYPE_LATEST)
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000   
    app.run(port=port)
