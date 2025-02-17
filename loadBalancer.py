from flask import Flask, request
import requests
import subprocess
import itertools
import os

loadBalancerPort = os.environ.get("PORT_LOAD_BALANCER",8080)
server1Port = os.environ.get("PORT_SERVER_1",8000)
server2Port = os.environ.get("PORT_SERVER_1",8001)
server3Port = os.environ.get("PORT_SERVER_1",8002)

app = Flask(__name__)

# List of backend server ports
backend_servers = [server1Port, server2Port, server3Port]
server_processes = []
request_count = 0  # Counter to implement round-robin

# Start backend servers
def start_server(port):
    return subprocess.Popen(["python", "app.py", str(port)])

# List of backend servers
SERVERS = [f"http://127.0.0.1:{server1Port}", f"http://127.0.0.1:{server2Port}", f"http://127.0.0.1:{server3Port}"]
server_pool = itertools.cycle(SERVERS)  # Round-robin rotation


# This Round Robin algorithm will be replaced by our Reinforcement learning agent
# The agent will take the action -> passing requests to the most optimal server. 

@app.route("/")
def hello():
    return "Ok!"

@app.route("/heavy-task")
def proxy_request():
    target_server = next(server_pool)  # Getting the next server
    try:
        response = requests.get(f"{target_server}/heavy-task")
        return response.text, response.status_code
    except requests.exceptions.RequestException as e:
        return f"Server {target_server} is down", 500

if __name__ == "__main__":
    
    # Start all servers
    for port in backend_servers:
        process = start_server(port)
        server_processes.append(process)

    app.run(port=loadBalancerPort)
