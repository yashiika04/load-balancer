from flask import Flask 
import requests
import subprocess
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import os 
import re
 
ALGORITHM = os.environ.get("RLAgent", "RoundRobin")  

print("Algorithm selected: ",ALGORITHM) 

if ALGORITHM == "RoundRobin":
    from loadBalancingAlgorithms.RoundRobin import select_optimal_server
elif ALGORITHM == "LeastConnection":
    from loadBalancingAlgorithms.LeastConnection import select_optimal_server
elif ALGORITHM == "RLAgent":
    from loadBalancingAlgorithms.RL_Agent import select_optimal_server
else:
    raise ValueError(f"Invalid Load Balancing Algorithm: {ALGORITHM}")
 
LOAD_BALANCER_PORT = int(os.environ.get("PORT_LOAD_BALANCER", 8005))
SERVER_1_PORT = int(os.environ.get("PORT_SERVER_1", 8000))
SERVER_2_PORT = int(os.environ.get("PORT_SERVER_2", 8001))
SERVER_3_PORT = int(os.environ.get("PORT_SERVER_3", 8002))

app = Flask(__name__) 
  
backend_servers = [SERVER_1_PORT, SERVER_2_PORT, SERVER_3_PORT]
server_processes = [] 
 
def start_server(port):
    return subprocess.Popen(["python", "app.py", str(port)])

# Backend servers
SERVERS = [
    f"http://127.0.0.1:{SERVER_1_PORT}",
    f"http://127.0.0.1:{SERVER_2_PORT}",
    f"http://127.0.0.1:{SERVER_3_PORT}",
]
 
server_pool = itertools.cycle(SERVERS)

def parse_metrics(metrics_text):
    """
    Parse Prometheus text for performance KPIs:
      - failed_requests: Count of 429 responses.
      - success_requests: Count of 200 responses.
      - avg_successful_response_time: computed as sum/count from histogram.
      - failed_to_success_ratio: ratio of failed to successful requests.
      - total_requests: sum of failed and successful requests.
    """
    # Extract count for failed responses (HTTP 429)
    failed_count_match = re.search(
        r'flask_http_request_total\{method="GET",status="429"\}\s+([\d.]+)', metrics_text)
    failed_count = float(failed_count_match.group(1)) if failed_count_match else 0.0

    # Extract count for successful responses (HTTP 200)
    success_count_match = re.search(
        r'flask_http_request_total\{method="GET",status="200"\}\s+([\d.]+)', metrics_text)
    success_count = float(success_count_match.group(1)) if success_count_match else 0.0

    # Extract sum for successful response times for /heavy-task
    success_sum_match = re.search(
        r'flask_http_request_duration_seconds_sum\{method="GET",path="/heavy-task",status="200"\}\s+([\d.]+)', metrics_text)
    success_sum = float(success_sum_match.group(1)) if success_sum_match else 0.0

    # Extract count for successful response times for /heavy-task
    success_time_count_match = re.search(
        r'flask_http_request_duration_seconds_count\{method="GET",path="/heavy-task",status="200"\}\s+([\d.]+)', metrics_text)
    success_time_count = float(success_time_count_match.group(1)) if success_time_count_match else 0.0

    # Compute average successful response time
    avg_success_response = success_sum / success_time_count if success_time_count > 0 else 1.0

    # Calculate failed-to-success ratio
    failed_to_success_ratio = failed_count / success_count if success_count > 0 else float('inf')
 
    total_requests = failed_count + success_count

    return {
        "failed_requests": failed_count,
        "success_requests": success_count,
        "avg_successful_response_time": avg_success_response,
        "failed_to_success_ratio": failed_to_success_ratio,
        "total_requests": total_requests
    } 
      

@app.route("/")
def hello():
    return "Load Balancer is Running!"


@app.route("/heavy-task")
def proxy_request():
    """
    Simulate traffic load balancing by forwarding requests to servers.
    Uses ThreadPoolExecutor to send multiple requests in parallel.
    """
    TOTAL_REQUESTS = 50
    responses = []

    with ThreadPoolExecutor(max_workers=TOTAL_REQUESTS) as executor:
        future_to_server = {}

        for i in range(TOTAL_REQUESTS):
            try:
                # Select an optimal server
                target_server = select_optimal_server(SERVERS)
                print(f"Selected server: {target_server}")
            except Exception as e:
                print(f"Algorithm failed, falling back to Round Robin. Error: {e}")
                target_server = next(server_pool)

            # Submit request to executor
            future = executor.submit(requests.get, f"{target_server}/heavy-task")
            future_to_server[future] = (target_server, i)

        # Collect responses
        for future in as_completed(future_to_server):
            target_server, call_id = future_to_server[future]
            try:
                response = future.result()
                if response.status_code == 200:
                    responses.append(f"API Call {call_id} → {target_server}: Success\n")
                else:
                    responses.append(f"API Call {call_id} → {target_server}: Failed with {response.status_code}\n")
            except Exception as e:
                responses.append(f"API Call {call_id} → {target_server}: Failed with {str(e)}\n")

    return "\n".join(responses)

   
@app.route("/server-metrics")
def fetch_server_metrics():
    metrics_data = {}
    for server in SERVERS:
        try:
             
            response = requests.get(f"{server}/metrics", timeout=5)
            response.raise_for_status()  # Raise an error for non-200 responses

            # Parse the raw Prometheus text into structured data
            parsed_data = parse_metrics(response.text)
            metrics_data[server] = {"metrics": parsed_data}

        except requests.exceptions.RequestException as e:
            metrics_data[server] = {"error": str(e)}

    return metrics_data

  
def calculate_reward(state):
    # A simple reward: negative sum of latencies (the lower, the better)
    return -sum(server["latency"] for server in state.values())
 
if __name__ == "__main__":

    # Starting all the servers
    for port in backend_servers:
        process = start_server(port)
        server_processes.append(process)

    app.run(host="0.0.0.0", port=LOAD_BALANCER_PORT)