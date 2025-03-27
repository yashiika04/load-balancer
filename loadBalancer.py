from flask import Flask
import requests
import subprocess
import itertools
import os
import numpy as np
from flask import jsonify, request 


# Load the algorithm dynamically based on environment variable
# ALGORITHM = os.environ.get("LB_ALGO", "RoundRobin")  # Default: Round Robin

ALGORITHM = "LeastConnection"

if ALGORITHM == "RoundRobin":
    from loadBalancingAlgorithms.RoundRobin import select_optimal_server
elif ALGORITHM == "LeastConnection":
    from loadBalancingAlgorithms.LeastConnection import select_optimal_server
# elif ALGORITHM == "RLAgent":
#     from loadBalancingAlgorithms.RL_Agent import select_optimal_server
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

@app.route("/")
def hello():
    return "Load Balancer is Running!"

@app.route("/heavy-task")
def proxy_request():
    """
    Forward request to the selected server using the chosen algorithm.
    If the selected algorithm fails, fallback to Round Robin.
    """
    try:
        target_server = select_optimal_server(SERVERS)
        print(f"Selected server: {target_server}")
    except Exception as e:
        print(f"Algorithm failed ({ALGORITHM}), falling back to Round Robin. Error: {e}")
        target_server = next(server_pool)

    try:
        response = requests.get(f"{target_server}/simulate-traffic")
        return response.text, response.status_code
    
    except requests.exceptions.RequestException:
        return f"Server {target_server} is down", 500

 # Simulated state for demonstration (replace with real metrics collection)

@app.route("/server-metrics")
def fetch_server_metrics():
    metrics_data = {}

    for server in SERVERS:
        try:
            response = requests.get(f"{server}/metrics")
            response.raise_for_status()  # Ensure HTTP 200

            # Instead of JSON, store raw Prometheus text format
            metrics_data[server] = {"metrics": response.text}

        except requests.exceptions.RequestException as e:
            metrics_data[server] = {"error": str(e)}

    return metrics_data

# for setting up environment for Load Balancer

current_state = {
    "server1": {"latency": np.random.uniform(0, 1), "cpu": np.random.uniform(0, 1)},
    "server2": {"latency": np.random.uniform(0, 1), "cpu": np.random.uniform(0, 1)},
    "server3": {"latency": np.random.uniform(0, 1), "cpu": np.random.uniform(0, 1)},
}

def calculate_reward(state):
    # A simple reward: negative sum of latencies (the lower, the better)
    return -sum(server["latency"] for server in state.values())

@app.route("/get_state", methods=["GET"])
def get_state():
    # Return the current state (you might aggregate metrics here)
    return jsonify(current_state)

@app.route("/apply_action", methods=["POST"])
def apply_action():
    data = request.json  # e.g., {"action": [0.2, 0.5, 0.3]}
    # Process the action here: update state based on action, etc.
    # For demonstration, let's simulate that the action affects the state:
    for i, server in enumerate(["server1", "server2", "server3"]):
        # A very simple dynamic update based on the action weight:
        current_state[server]["latency"] = np.clip(current_state[server]["latency"] - 0.1 * data["action"][i] + np.random.uniform(0, 0.05), 0, 1)
    
    reward = calculate_reward(current_state)
    return jsonify({"reward": reward, "new_state": current_state})

if __name__ == "__main__":

    # Starting all the servers
    for port in backend_servers:
        process = start_server(port)
        server_processes.append(process)

    app.run(host="0.0.0.0", port=LOAD_BALANCER_PORT)