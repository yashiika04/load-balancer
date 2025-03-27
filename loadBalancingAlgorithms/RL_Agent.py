import tensorflow as tf
import numpy as np
import random
 
policy_dir = "saved_policies/load_balancing_trained_policy"

try:
    agent = tf.compat.v2.saved_model.load(policy_dir)    
    use_rl_model = True
except Exception as e:
    print(f"Failed to load RL model: {e}")
    use_rl_model = False

def preprocess_state(server_metrics):
    """
    Convert server metrics into a format suitable for RL model input.
    Example: Normalize CPU and latency values.
    """
    return np.array([server_metrics["cpu"], server_metrics["latency"]], dtype=np.float32)

def select_optimal_server(servers, server_metrics):
    """
    Selects the optimal server using an RL model.
    If the model is unavailable, falls back to random selection.
    
    Args:
    - servers: List of available servers.
    - server_metrics: Dictionary containing CPU and latency data for each server.

    Returns:
    - Selected server (string)
    """
    if use_rl_model:
        try:
            # Convert the server metrics into RL-compatible input
            state = np.array([preprocess_state(server_metrics[srv]) for srv in servers])
            state = tf.convert_to_tensor(state)

            # Get the best action from the RL model
            action = agent.policy.action(state)
            selected_server = servers[int(action.numpy())]

            return selected_server
        except Exception as e:
            print(f"RL model inference failed: {e}")

    # Fallback to random selection if model fails
    return random.choice(servers)
