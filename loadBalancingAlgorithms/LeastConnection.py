import threading
import random

class LeastConnections:
    def __init__(self, servers):
        self.servers = [{"url": url, "connections": random.randint(0, 5)} for url in servers]
        self.lock = threading.Lock()

    def get_least_connections_server(self):
        with self.lock:
            return min(self.servers, key=lambda server: server["connections"])

    def select_optimal_server(self, servers):
        self.servers = [{"url": url, "connections": random.randint(0, 5)} for url in servers]
        return self.get_least_connections_server()["url"]

least_connection_lb = LeastConnections([])

def select_optimal_server(servers):
    return least_connection_lb.select_optimal_server(servers)
