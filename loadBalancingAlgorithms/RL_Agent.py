import os
import tensorflow as tf 
from tf_agents.environments import tf_py_environment  
from tf_agents.environments import py_environment  
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import time
import numpy as np
import random
import requests
import re

POLICY_DIRS = {
    "model1": "loadBalancingAlgorithms/saved_policies/load_balancing_trained_policy_model1",
    "model2": "loadBalancingAlgorithms/saved_policies/load_balancing_trained_policy_model2",
    "model3": "loadBalancingAlgorithms/saved_policies/load_balancing_trained_policy_model3",
    "model4": "loadBalancingAlgorithms/saved_policies/load_balancing_trained_policy_model4",
    "model5": "loadBalancingAlgorithms/saved_policies/load_balancing_trained_policy_model5",
}

def get_reward_mode() -> str:
    mode = os.getenv("RL_REWARD_MODE", "model1").strip().lower()
    if mode not in POLICY_DIRS:
        print(f"Warning: RL_REWARD_MODE='{mode}' is not recognized. Falling back to model1.")
        return "model1"
    return mode


def get_policy_dir() -> str:
    env_policy_dir = os.getenv("RL_POLICY_DIR", "").strip()
    if env_policy_dir:
        print(f"Using RL_POLICY_DIR override: {env_policy_dir}")
        return env_policy_dir
    mode = get_reward_mode()
    policy_dir = POLICY_DIRS[mode]
    print(f"Using default policy_dir for reward mode '{mode}': {policy_dir}")
    return policy_dir

REWARD_MODE = get_reward_mode()
policy_dir = get_policy_dir()
_num_servers = 3


global SERVERS
SERVERS = [
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8002"
]

MAX_LATENCY   = 5.0     # seconds
MAX_REQUESTS  = 10000.0

try:
    # Load the saved policy from disk. 
    agent = tf.compat.v2.saved_model.load(policy_dir)
    use_rl_model = True
    print(f"RL model loaded successfully: {policy_dir}")
except Exception as e:
    print(f"Failed to load RL model: {e}")
    use_rl_model = False


_action_history = []   # module-level rolling window, reset each episode

def reset_action_history():
    global _action_history
    _action_history = []

def compute_reward_from_state(state_row, chosen_index = -1, alpha=1.0, beta=1.0, gamma=1.0,  lam=0.5, sla_norm=0.1, n_servers=3, history_window=10, mode=None):
   
    global _action_history

    latency = float(state_row[0])
    total_requests = float(state_row[1])
    failure_ratio = float(state_row[2])

    # Compute throughput as the effective successful request ratio times total requests.
    throughput = total_requests * max(0.0, 1.0 - failure_ratio)

    if mode is None:
        mode = REWARD_MODE

    mode = mode.lower()

    if mode == "model1":
        reward = - (alpha * latency + beta * failure_ratio) + gamma * throughput
    elif mode == "model3":
        SLA_THRESHOLD = 0.3 
        latency_penalty = max(0, latency - SLA_THRESHOLD)
        reward = -(alpha * latency_penalty + beta * failure_ratio)

    elif mode == "model4":
        violation   = max(0.0, latency - sla_norm)
        latency_pen = alpha * (violation ** 2)

        failure_pen = beta * failure_ratio
        _action_history.append(chosen_index)
        if len(_action_history) > history_window:
            _action_history.pop(0)

        counts = np.bincount(_action_history, minlength=n_servers).astype(np.float32)
        probs  = counts / counts.sum()
        eps    = 1e-8
        H      = -np.sum(probs * np.log(probs + eps))          
        H_max  = np.log(n_servers)                           
        fairness_bon = lam * (H / H_max)       

        reward = -latency_pen - failure_pen + fairness_bon
        reward = float(reward)

    elif mode == "model2":
        success_reward = 0.2 * (1.0 - failure_ratio)

        violation   = max(0.0, latency - sla_norm)
        latency_pen = alpha * violation

        failure_pen = beta * failure_ratio

        _action_history.append(chosen_index)
        if len(_action_history) > history_window:
            _action_history.pop(0)
       
        counts = np.bincount(_action_history, minlength=n_servers).astype(np.float32)
        probs  = counts / counts.sum()
        eps    = 1e-8
        H      = -np.sum(probs * np.log(probs + eps))
        H_max  = np.log(n_servers)
        fairness_bon = lam * (H / H_max)

        reward = success_reward - latency_pen - failure_pen + fairness_bon
        reward = float(reward)
    else:
        raise ValueError(f"Unsupported reward mode: {mode}")

    return reward

def _parse_prometheus_metrics(metrics_text: str) -> dict:
    """Extract aggregated metrics from Prometheus text format."""
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

    success_time_count_match = re.search(
        r'flask_http_request_duration_seconds_count\{method="GET",path="/heavy-task",status="200"\}\s+([\d.]+)', metrics_text)
    success_time_count = float(success_time_count_match.group(1)) if success_time_count_match else 0.0

    avg_success_response = success_sum / success_time_count if success_time_count > 0 else 0.0

    failed_to_success_ratio = failed_count / success_count if success_count > 0 else (float('inf') if total_requests > 0 else 0.5)

    return {
        "avg_successful_response_time": avg_success_response,
        "total_requests": total_requests,
        "failed_to_success_ratio": failed_to_success_ratio
    }


def _generate_state():
    """
    Fetch performance metrics from backend servers directly.
    For each server, extract:
      - avg_successful_response_time (latency)
      - total_requests
      - failed_to_success_ratio
    Returns a (num_servers, 3) numpy array.
    """
    latency = []
    requests_handled = []
    failed_to_success_ratio = []

    for server in SERVERS:
        try:
            response = requests.get(f"{server}/metrics", timeout=2)
            response.raise_for_status()
            backend_metrics = _parse_prometheus_metrics(response.text)

            print(f"{server}: {backend_metrics}")

            raw_latency = backend_metrics.get("avg_successful_response_time", 0.0)
            latency.append(min(raw_latency / 5.0, 1.0))
            raw_requests = backend_metrics.get("total_requests", 0.0)
            requests_handled.append(min(raw_requests / 10000, 1.0))
            failed_to_success_ratio.append(backend_metrics.get("failed_to_success_ratio", 0.5))
        except requests.RequestException as e:
            print(f"Error fetching metrics from {server}: {e}")
            latency.append(1.0)
            requests_handled.append(0.0)
            failed_to_success_ratio.append(0.5)

    state = np.column_stack([latency, requests_handled, failed_to_success_ratio])

    print("State :", state)

    return state.astype(np.float32)

class LoadBalancerEnv(py_environment.PyEnvironment):

    def __init__(self, servers, reward_mode=None):
        """
        Args:
          servers: List of server URLs
          reward_mode: Reward scheme to use ('full', 'latency', 'sla').
        """
        super(LoadBalancerEnv, self).__init__()
        self._servers = servers
        self._num_servers = len(servers)
        self._reward_mode = reward_mode or REWARD_MODE

        # Observation: For each server, we consider 3 metrics:
        # [avg_successful_response_time (latency), total_requests, failed_to_success_ratio]
        if self._reward_mode == "model5":
            self._observation_spec = array_spec.BoundedArraySpec(
                shape=(self._num_servers * 3,),   # flat: 9 values
                dtype=np.float32,
                minimum=np.array([0.0]*9, dtype=np.float32),
                maximum=np.array([1.0, 1.0, 5.0] * self._num_servers, dtype=np.float32),  # failure_ratio unbounded above 1
                name='observation'
            )
        else:
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
        self._max_steps = 100  # defines the episode length

        # Initialize state by fetching the metrics
        self._state = _generate_state()

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode_ended = False
        self._step_count = 0
        reset_action_history()          # clear entropy window each episode
        self._state = _generate_state()
        return ts.restart(self._state)

    def _step(self, action):

        if self._episode_ended:
            return self.reset()

        self._step_count += 1

        # Discrete action: chosen server index (0, 1, or 2)
        chosen_index = int(action)
        chosen_server = self._servers[chosen_index]
        print(f"Routing new traffic to chosen server: {chosen_server}")
       
        new_state = _generate_state() 

        start = chosen_index * 3
        
        if self._reward_mode == "model2":
            reward = compute_reward_from_state(
            state_row      = new_state[start:start+3],
            chosen_index   = chosen_index,
            alpha          = 1.5,
            beta           = 2.0,
            lam            = 0.3,
            sla_norm       = 0.1,
            n_servers      = self._num_servers,
            history_window = 50,
            mode=self._reward_mode
        )
        elif self._reward_mode == "model4":
            reward = compute_reward_from_state(
            state_row    = new_state[chosen_index],
            chosen_index = chosen_index,
            alpha        = 0.5,  
            beta         = 0.3,  
            lam          = 0.7,   
            sla_norm     = 0.1,  
            n_servers    = self._num_servers,
            history_window = 20,
            mode=self._reward_mode
        )
        else:
            reward = compute_reward_from_state(
                new_state[chosen_index],
                mode=self._reward_mode
            )

        print(f"Reward: {reward}")
 
        self._state = new_state

        if self._step_count >= self._max_steps:
            self._episode_ended = True
            return ts.termination(self._state, reward)
        else:
            return ts.transition(self._state, reward=reward, discount=0.99)
 

class RLBasedLoadBalancer:
    def __init__(self, servers, policy_dir):
        self.servers = servers
        self._num_servers = len(servers)
        self.use_rl_model = False

        try:
            self.agent = tf.compat.v2.saved_model.load(policy_dir)
            self.env = tf_py_environment.TFPyEnvironment(
                LoadBalancerEnv(servers, reward_mode=REWARD_MODE)
            )
            self.use_rl_model = True
            print(f"RL model and environment initialized using policy_dir={policy_dir} reward_mode={REWARD_MODE}.")
        except Exception as e:
            print(f"Failed to initialize RL model: {e}")

    def select_optimal_server(self):
        epsilon = 0.1  # 10% random exploration
        if random.random() < epsilon:
            chosen = random.choice(self.servers)
            print(f"Exploration: randomly chosen server {chosen}")
            return chosen
        
        if not self.use_rl_model:
            print("RL model unavailable, will run fallback loop")
        else:

            try:
                print("---- RL DEBUG START ----")
                print("Agent object:", self.agent)
                print("Available signatures:", getattr(self.agent, "signatures", None))

                time_step = self.env.reset()

                print("Time step type:", type(time_step))
                print("Time step observation shape:", time_step.observation.shape)
                print("Time step observation:", time_step.observation)


                action_tensor = self.agent.action(time_step)   # returns raw int32 tensor shape (1,)
                action_index = int(action_tensor.numpy()[0])

                # action_step = self.agent.action(time_step)
                # print("Action step:", action_step)
                # action = action_step.action.numpy()[0]
                # action_index = int(action)

                print("Chosen action index:", action_index)

                selected_server = self.servers[action_index]
        
                print(f"RL agent suggested index {action_index} -> {selected_server}")


                try:
                    response = requests.get(selected_server, timeout=2)
                    print(f"Health check for RL choice {selected_server}: {response.status_code}")
                    if response.status_code == 200:
                        print(f"RL agent selected server index: {action_index}")
                        return selected_server
                    else:
                        print(f"RL-selected server unhealthy: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"Health check request to RL choice failed: {e}")

            except Exception as e:
                print(f"Exception during RL decision: {e}")
            
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
        print(f"Fallback picking random server {fallback} (all health checks failed)")
        return fallback
