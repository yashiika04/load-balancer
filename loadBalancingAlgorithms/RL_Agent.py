import tensorflow as tf 
from tf_agents.environments import tf_py_environment  
from tf_agents.environments import py_environment  
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import numpy as np
import random
import requests

# Load the trained RL policy   
policy_dir = "loadBalancingAlgorithms/saved_policies/load_balancing_trained_policy"

serverMetricsUrl = "http://localhost:8005/server-metrics"
_num_servers = 3

global SERVERS
SERVERS = [
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8002"
]

try:
    # Load the saved policy from disk. 
    agent = tf.compat.v2.saved_model.load(policy_dir)
    use_rl_model = True
    print("RL model loaded successfully.")
except Exception as e:
    print(f"Failed to load RL model: {e}")
    use_rl_model = False


def compute_reward_from_state(state_row, alpha=1.0, beta=1.0, gamma=1.0):
    """
    Compute a reward from a state row (NumPy array) representing a server's metrics.
    
    Args:
      state_row (numpy.array): Array with [latency, total_requests, failed_to_success_ratio].
      alpha (float): Weight for latency penalty.
      beta (float): Weight for failure ratio penalty.
      gamma (float): Weight for throughput reward.
      
    Returns:
      float: Computed reward.
    """
    latency = state_row[0]
    total_requests = state_row[1]
    failure_ratio = state_row[2]
    
    # Compute throughput as an example: the effective successful request ratio times total requests.
    throughput = total_requests * (1 - failure_ratio)
    
    # Reward: lower latency and lower failure ratio are good; higher throughput is good.
    reward = - (alpha * latency + beta * failure_ratio) + gamma * throughput
    
    return reward

def _generate_state():
    """
    Fetch performance metrics from the aggregated metrics endpoint.
    For each server, extract:
      - avg_successful_response_time (latency)
      - total_requests
      - failed_to_success_ratio
    Returns a (num_servers, 3) numpy array.
    """
    latency = []
    requests_handled = []
    failed_to_success_ratio = []
    
    try:
        response = requests.get(serverMetricsUrl)
        response.raise_for_status()
        server_metrics = response.json()
         
        for server in SERVERS:
            data = server_metrics.get(server, {})
            if "metrics" in data and data["metrics"] is not None:
                metrics = data["metrics"]
                
                print(f"{server}: {metrics}") 

                latency.append(metrics.get("avg_successful_response_time", 1.0))
                requests_handled.append(metrics.get("total_requests", 0.0))
                failed_to_success_ratio.append(metrics.get("failed_to_success_ratio", 0.5))
            else: 
                latency.append(2.0)
                requests_handled.append(0.0)
                failed_to_success_ratio.append(0.1)

    except requests.RequestException as e:
        print(f"Error fetching server metrics: {e}")
        latency = [1.0] * _num_servers
        requests_handled = [0.0] * _num_servers
        failed_to_success_ratio = [0.5] * _num_servers
    
    state = np.column_stack([latency, requests_handled, failed_to_success_ratio])

    print("State :", state)

    return state.astype(np.float32)

class LoadBalancerEnv(py_environment.PyEnvironment):

    def __init__(self, servers):
        """
        Args:
          serverMetricsUrl: URL that returns aggregated server metrics in JSON format.
          servers: List of server URLs 
        """
        super(LoadBalancerEnv, self).__init__()
        self._servers = servers
        self._num_servers = len(servers) 

        # Observation: For each server, we consider 3 metrics:
        # [avg_successful_response_time (latency), total_requests, failed_to_success_ratio]
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._num_servers, 3),
            dtype=np.float32,
            minimum=0.0,
            maximum=1.0,
            name='observation'
        )

        # Action: A single discrete action representing the server index (0 to num_servers-1).
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=self._num_servers - 1,
            name='action'
        )

        self._episode_ended = False
        self._step_count = 0
        self._max_steps = 50  # defines the episode length

        # Initialize state by fetching the metrics
        self._state = _generate_state()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self._step_count = 0
        self._state = _generate_state()
        return ts.restart(self._state)

    def _step(self, action):

        if self._episode_ended:
            return self.reset()

        self._step_count += 1

        # Discrete action: chosen server index (0, 1, or 2)
        chosen_index = int(action)
       
        new_state = _generate_state() 
         
        reward = compute_reward_from_state(new_state[chosen_index])

        # print(f"Reward: {reward}")
 
        self._state = new_state

        if self._step_count >= self._max_steps:
            self._episode_ended = True
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward=reward, discount=0.99)
 

class RLBasedLoadBalancer:
    def __init__(self, servers, policy_dir, metrics_url):
        self.servers = servers
        self.serverMetricsUrl = metrics_url
        self._num_servers = len(servers)
        self.use_rl_model = False

        try:
            self.agent = tf.compat.v2.saved_model.load(policy_dir)
            self.env = tf_py_environment.TFPyEnvironment(LoadBalancerEnv(servers))
            self.use_rl_model = True
            print("RL model and environment initialized.")
        except Exception as e:
            print(f"Failed to initialize RL model: {e}")

    def select_optimal_server(self):
        epsilon = 0.2  # 20% random exploration
        if random.random() < epsilon:
            return random.choice(self.servers)
    
        if self.use_rl_model:
            try:
                time_step = self.env.reset()
                action_step = self.agent.action(time_step)
                action_index = int(action_step.action.numpy())
                selected_server = self.servers[action_index]

                try:
                    response = requests.get(selected_server, timeout=3)
                    if response.status_code == 200:
                        print(f"RL agent selected server index: {action_index}")
                        return selected_server
                    else:
                        print(f"RL-selected server unhealthy: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print()

            except Exception as e:
                print()

        # Fallback
        for server in random.sample(self.servers, len(self.servers)):
            try:
                response = requests.get(server, timeout=3)
                if response.status_code == 200:
                    print(f"Fallback selected healthy server: {server}")
                    return server
            except requests.exceptions.RequestException:
                continue

        fallback = random.choice(self.servers)
        return fallback
