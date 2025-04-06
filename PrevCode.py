

loadBalancerEnv = LoadBalancerEnv(SERVERS)
tf_env = tf_py_environment.TFPyEnvironment(loadBalancerEnv)



def select_optimal_server(servers):
    """
    Select the optimal server using an RL model. If inference fails, fall back to a live server randomly.
    """
    if use_rl_model:
        try:
 
            loadBalancerEnv = LoadBalancerEnv(servers)
            tf_env = tf_py_environment.TFPyEnvironment(loadBalancerEnv)
            
            # Get the initial state
            time_step = tf_env.reset()

            # Ask the agent for an action
            action_step = agent.action(time_step)

            # Extract the scalar index from the Tensor
            action_index = int(action_step.action.numpy())
            selected_server = servers[action_index]
 
            health_url = selected_server

            try:
                response = requests.get(health_url, timeout=3)
                if response.status_code == 200:
                    print(f"RL agent selected server index: {action_index}")
                    return selected_server
                else:
                    print(f"RL-selected server unhealthy: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Health check failed for RL-selected server: {e}")

        except Exception as e:
            print(f"RL model inference failed: {e}")
  
    for server in random.sample(servers, len(servers)):
        try:
            response = requests.get(f"{server}", timeout=3)
            if response.status_code == 200:
                print(f"Fallback selected healthy server: {server}")
                return server
        except requests.exceptions.RequestException:
            continue
 
    selected = random.choice(servers)
    print(f"No healthy servers found. Returning random server: {selected}")
    return selected



def select_optimal_server2(servers):
    """
    Select the optimal server using an RL model. If inference fails, fall back to random.
    """ 
    if use_rl_model:
        try:

            loadBalancerEnv = LoadBalancerEnv(SERVERS)
            train_env = tf_py_environment.TFPyEnvironment(loadBalancerEnv)
        
            time_step = train_env.reset() 

            action_step = agent.action(time_step)
            # print(action_step)

            time_step = train_env._step(action_step.action)

            # print(time_step)
            
            print("Action taken:", action_step.action.numpy())  

            return servers[action_step.action.numpy()[0]]
        except Exception as e:
            print(f"RL model inference failed: {e}")

    # Fallback
    selected = random.choice(servers)
    
    print(f"Falling back to random selection: {selected}")

    return selected
