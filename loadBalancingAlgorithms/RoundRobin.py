import itertools
 
_server_cycle = None

def initialize(servers): 
    global _server_cycle
    _server_cycle = itertools.cycle(servers)

def select_optimal_server(servers):
    """
    Return the next server in a round-robin fashion.
    If the cycle is not yet initialized, it will be set up.
    """
    global _server_cycle
    if _server_cycle is None:
        _server_cycle = itertools.cycle(servers)
    return next(_server_cycle)
