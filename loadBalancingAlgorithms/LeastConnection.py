import threading
import random

class LeastConnectionLoadBalancer:
    def __init__(self, servers):
        self.lock = threading.Lock()
        self.servers = [{"url": url, "connections": random.randint(0, 5)} for url in servers]

    def get_least_connections_server(self):
        with self.lock:
            return min(self.servers, key=lambda server: server["connections"])

    def select_optimal_server(self):
        # Simulate updated connection count for demo purposes
        for server in self.servers:
            server["connections"] = random.randint(0, 5)
        return self.get_least_connections_server()["url"]
