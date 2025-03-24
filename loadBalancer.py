from flask import Flask 
import requests
import subprocess
import itertools
import os
 
# from loadBalancingAlgorithms.LeastConnection import select_optimal_server as load_balancing_algo 
from loadBalancingAlgorithms.RoundRobin import select_optimal_server as load_balancing_algo

LOAD_BALANCER_PORT = os.environ.get("PORT_LOAD_BALANCER",3000)
SERVER_1_PORT = os.environ.get("PORT_SERVER_1",8000)
SERVER_2_PORT = os.environ.get("PORT_SERVER_2",8001)
SERVER_3_PORT = os.environ.get("PORT_SERVER_3",8002)

app = Flask(__name__)
 
backend_servers = [SERVER_1_PORT, SERVER_2_PORT, SERVER_3_PORT]
server_processes = [] 
 
def start_server(port):
    return subprocess.Popen(["python", "app.py", str(port)])

# List of backend servers
SERVERS = [f"http://127.0.0.1:{SERVER_1_PORT}", f"http://127.0.0.1:{SERVER_2_PORT}", f"http://127.0.0.1:{SERVER_3_PORT}"]
server_pool = itertools.cycle(SERVERS)  # Round-robin rotation


# This Round Robin algorithm will be replaced by our Reinforcement learning agent
# The agent will take the action -> passing requests to the most optimal server,
# in order to maximize throughput and minimize latency.

@app.route("/")
def hello():
    return "Ok!"

@app.route("/heavy-task")
def proxy_request():
    try: 
        target_server = load_balancing_algo(SERVERS)
        print("Exception didn't occur !")
    except Exception as e:
        # If the chosen algorithm fails, fall back to round-robin
        print("Exception occured!")
        target_server = next(server_pool)
    
    try:
        response = requests.get(f"{target_server}/heavy-task")
        return response.text, response.status_code
    except requests.exceptions.RequestException:
        return f"Server {target_server} is down", 500

if __name__ == "__main__":
    
    # Starting all the servers
    for port in backend_servers:
        process = start_server(port)
        server_processes.append(process)

    app.run(host="0.0.0.0", port=LOAD_BALANCER_PORT) 

