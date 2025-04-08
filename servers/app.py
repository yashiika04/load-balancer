import time
import random
import sys 

from flask import Flask, jsonify, Response, request
from flask_limiter import Limiter  
from flask_limiter.util import get_remote_address    
 
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from prometheus_client import Counter, Histogram, generate_latest

from prometheus_flask_exporter import PrometheusMetrics 

app = Flask(__name__)
limiter = Limiter(get_remote_address, app=app, default_limits=["5 per second"])   

global SERVER_PORT
SERVER_PORT = 8000


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
    """Simulate a heavy computation with a random delay"""
    delay = random.uniform(0.5, 2)  # Delay between 0.5s and 2s
    time.sleep(delay)
    return "OK!!"

# --- Endpoints ---
 

@app.route("/")
def index():
    return jsonify({"message": "Server is running!"})

# Server capacity limits
server_capacity = {
    8000: 8,   
    8001: 15,   
    8002: 16   
}

SERVER_PORT = 8000  

# Use the rate limiter to limit requests based on server capacity
@app.route("/heavy-task")
@limiter.limit(lambda: f"{server_capacity.get(SERVER_PORT, 5)} per second")   
def heavy_task():
    try:
        result = heavyOperation()
        return result
    except Exception as e:
        return str(e), 500

  
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
    SERVER_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000   
    app.run(host="0.0.0.0", port=SERVER_PORT)
 
