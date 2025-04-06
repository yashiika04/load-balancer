import itertools
import requests
import random

class RoundRobinLoadBalancer:
    def __init__(self, servers):
        self.servers = servers
        self._server_cycle = itertools.cycle(servers)
    
    def select_optimal_server(self):
        """
        Return the next healthy server in round-robin order.
        If no healthy server is found, fall back to a random one.
        """
        for _ in range(len(self.servers)):
            server = next(self._server_cycle)
            try:
                response = requests.get(server, timeout=3)
                if response.status_code == 200:
                    print(f"Round-robin selected server: {server}")
                    return server
            except requests.exceptions.RequestException:
                continue

        fallback = random.choice(self.servers)
        
        print(f"No healthy servers found in round-robin. Returning random server: {fallback}")
        return fallback
