from flask import Flask, jsonify
import requests 
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from dotenv import load_dotenv 
import re
import time

from loadBalancingAlgorithms.RoundRobin import RoundRobinLoadBalancer
from loadBalancingAlgorithms.LeastConnection import LeastConnectionLoadBalancer
from loadBalancingAlgorithms.RL_Agent import RLBasedLoadBalancer
 
load_dotenv(override=True)  
 
LB_ALGORITHM = os.getenv("LB_ALGO")  

print("Algorithm selected: ",LB_ALGORITHM) 
 
 
LOAD_BALANCER_PORT = int(os.environ.get("PORT_LOAD_BALANCER")) # type: ignore
SERVER_1_PORT = int(os.environ.get("PORT_SERVER_1")) # type: ignore
SERVER_2_PORT = int(os.environ.get("PORT_SERVER_2")) # type: ignore
SERVER_3_PORT = int(os.environ.get("PORT_SERVER_3")) # type: ignore

app = Flask(__name__) 
  
backend_servers = [SERVER_1_PORT, SERVER_2_PORT, SERVER_3_PORT]
server_processes = [] 
   
# Backend servers
SERVERS = [
    f"http://127.0.0.1:{SERVER_1_PORT}",
    f"http://127.0.0.1:{SERVER_2_PORT}",
    f"http://127.0.0.1:{SERVER_3_PORT}",
]
 
server_pool = itertools.cycle(SERVERS)

policy_dir = "loadBalancingAlgorithms/saved_policies/load_balancing_trained_policy1"
serverMetricsUrl = "http://localhost:8005/server-metrics"

class LoadBalancer:
    def __init__(self, servers):
        self.servers = servers

        if LB_ALGORITHM == "RoundRobin":
            self.strategy = RoundRobinLoadBalancer(servers)
        elif LB_ALGORITHM == "LeastConnection":
            self.strategy = LeastConnectionLoadBalancer(servers)
        elif LB_ALGORITHM == "RLAgent":
            self.strategy = RLBasedLoadBalancer(servers, policy_dir)
        else:
            raise ValueError(f"Invalid Load Balancing Algorithm: {LB_ALGORITHM}")

    def select_optimal_server(self):
        return self.strategy.select_optimal_server()

 
lb = LoadBalancer(SERVERS)
 
def parse_metrics(metrics_text):
    """
    Parse Prometheus text for performance KPIs:
      - failed_requests: Count of 429 responses.
      - success_requests: Count of 200 responses.
      - avg_successful_response_time: computed as sum/count from histogram.
      - failed_to_success_ratio: ratio of failed to successful requests.
      - total_requests: sum of failed and successful requests.
    """ 
    failed_count_match = re.search(
        r'flask_http_request_total\{method="GET",status="429"\}\s+([\d.]+)', metrics_text)
    failed_count = float(failed_count_match.group(1)) if failed_count_match else 0.0
 
    success_count_match = re.search(
        r'flask_http_request_total\{method="GET",status="200"\}\s+([\d.]+)', metrics_text)
    success_count = float(success_count_match.group(1)) if success_count_match else 0.0

    total_requests = failed_count + success_count
 
    success_sum_match = re.search(
        r'flask_http_request_duration_seconds_sum\{method="GET",path="/heavy-task",status="200"\}\s+([\d.]+)', metrics_text)
    success_sum = float(success_sum_match.group(1)) if success_sum_match else 0.0

    # Extract count for successful response times for /heavy-task
    success_time_count_match = re.search(
        r'flask_http_request_duration_seconds_count\{method="GET",path="/heavy-task",status="200"\}\s+([\d.]+)', metrics_text)
    success_time_count = float(success_time_count_match.group(1)) if success_time_count_match else 0.0
 
    avg_success_response = success_sum / success_time_count if success_time_count > 0 else 1.0
 
    # failed_to_success_ratio = failed_count / success_count if success_count > 0 else float('inf')
    failed_to_success_ratio = failed_count / success_count if success_count > 0 else (float('inf') if total_requests > 0 else 0.5)
 
    

    return {
        "failed_requests": failed_count,
        "success_requests": success_count,
        "avg_successful_response_time": avg_success_response,
        "failed_to_success_ratio": failed_to_success_ratio,
        "total_requests": total_requests
    } 
      
@app.route("/")
def home():
    return jsonify({
        "service": "Load Balancer",
        "algorithm": LB_ALGORITHM,
        "available_routes": [
            "/health-check",
            "/heavy-task",
            "/server-metrics"
        ]
    })


@app.route("/health-check")
def health_check():
    return jsonify({
        "status": "OK",
        "message": "Load Balancer is Running!",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    

@app.route("/heavy-task")
def proxy_request():

    """
    Simulate traffic load balancing by forwarding requests to servers.
    Uses ThreadPoolExecutor to send multiple requests in parallel.
    """
    
    TOTAL_REQUESTS = 500
    successes = 0
    failures = 0
    details = []

    with ThreadPoolExecutor(max_workers=150) as executor:
        future_to_info = {}
        for i in range(TOTAL_REQUESTS):
            try:
                target_server = lb.select_optimal_server()
                print(f"Selected server: {target_server}")
            except Exception as e:
                print(f"Algorithm failed, falling back to Round Robin. Error: {e}")
                target_server = next(server_pool)
            future = executor.submit(requests.get, f"{target_server}/heavy-task", timeout=5)
            future_to_info[future] = {"server": target_server, "call_id": i}
        
        for future in as_completed(future_to_info):
            info = future_to_info[future]
            try:
                response = future.result()
                if response.status_code == 200:
                    successes += 1
                    info["status"] = "Success"
                else:
                    failures += 1
                    info["status"] = f"Failed with {response.status_code}"
            except Exception as e:
                failures += 1
                info["status"] = f"Failed with {str(e)}"
            details.append(info)

    result = {
        "total_requests": TOTAL_REQUESTS,
        "successes": successes,
        "failures": failures,
        "details": details
    }
    return jsonify(result)


@app.route("/server-metrics")
def fetch_server_metrics():
    metrics_data = {}
    for server in SERVERS:
        try:
            response = requests.get(f"{server}/metrics", timeout=2)
            response.raise_for_status()   
 
            parsed_data = parse_metrics(response.text)
            metrics_data[server] = {"metrics": parsed_data}

        except requests.exceptions.RequestException as e: 
 
            fallback_metrics = {
                "avg_successful_response_time": 10.0,   
                "total_requests": 100.0,            
                "failed_to_success_ratio": 1.0      
            }

            metrics_data[server] = {
                "metrics": fallback_metrics,
                "error": str(e)
            }

    return metrics_data

 
if __name__ == "__main__":  
    app.run(host="0.0.0.0", port=LOAD_BALANCER_PORT, threaded=True)