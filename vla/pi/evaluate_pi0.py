#!/usr/bin/env python3

# Adapted from https://github.com/huggingface/lerobot/blob/main/examples/lekiwi/evaluate.py
# And other examples from LeRobot repo
# LeRobot and OpenPi should be installed
# Run `python evaluate_pi0.py` to run the model

import os
# Configure JAX memory allocation before importing JAX-related modules
# Is needed to resolve possible issues with memory allocation
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import time
import numpy as np

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.robot_utils import busy_wait

# Import Pi0 model from openpi
from openpi.training import config as pi0_config
from openpi.policies import policy_config
from huggingface_hub import snapshot_download

# Configuration
FPS = 30
TASK_DESCRIPTION = "Pick the orange ball and put it in the green cup"
ACTIONS_TO_EXECUTE = 20  # Execute this many actions from each predicted chunk

# Load Pi0 model
print("Loading Pi0 model...")

# Use your robot Pi0 config - in my case "pi0_lekiwi_fast", "pi0_lekiwi" or "pi05_lekiwi"
config = pi0_config.get_config("pi0_lekiwi")

# Use you trained policy HF directory
# You should upload your model (assets and params directories from the checkpoint) to Hugging Face to use it here
checkpoint_dir = snapshot_download(repo_id="YOUR_HF_REPO_PATH")
pi0_policy = policy_config.create_trained_policy(config, checkpoint_dir)
print("Pi0 model loaded successfully")

# Connect to robot
# Use your robot IP and ID
robot_config = LeKiwiClientConfig(remote_ip="YOUR_ROBOT_IP", id="YOUR_ROBOT_ID")
robot = LeKiwiClient(robot_config)
robot.connect()

if not robot.is_connected:
    raise ValueError("Robot is not connected!")

print(f"Robot connected. Starting control loop with task: '{TASK_DESCRIPTION}'")
print(f"Will execute {ACTIONS_TO_EXECUTE} actions from each predicted chunk")

# Control loop variables
step = 0
last_actions = None
action_index = 0
pred_times = []

while True:
    t0 = time.perf_counter()
    
    # Run prediction when we need new actions (either first time or when we've executed enough actions)
    if last_actions is None or action_index >= min(ACTIONS_TO_EXECUTE, len(last_actions)):
        # Get robot observation
        observation = robot.get_observation()

        # Prepare input for Pi0
        # Adjust for your robot
        pi0_input = {
            "prompt": TASK_DESCRIPTION,
            "observation/state": observation["observation.state"],
        }
        
        for k in ["top", "wrist", "front"]:
            pi0_input[f"observation/images/{k}"] = observation[k]
        
        # Run inference
        t_pred_start = time.perf_counter()
        output = pi0_policy.infer(pi0_input)
        t_pred = time.perf_counter() - t_pred_start
        
        # Keep track of last 10 prediction times
        pred_times.append(t_pred)
        pred_times = pred_times[-10:]  # Keep last 10
        
        last_actions = output["actions"]
        action_index = 0

        print(f"Step {step}: Predicted {len(last_actions)} actions, will execute {min(ACTIONS_TO_EXECUTE, len(last_actions))}")
        print(f"Prediction took {t_pred:.3f}s (avg over last {len(pred_times)}: {np.mean(pred_times):.3f}s)")
    
    # Execute action
    else:
        # Get current action from the sequence
        action = last_actions[action_index]
        
        # Convert action array to robot's expected format        
        action_dict = {}
        for i, action_name in enumerate(list(robot.action_features.keys())):
            if i < len(action):
                action_dict[action_name] = action[i]
        
        robot.send_action(action_dict)
        action_index += 1
        step += 1
    
    # Maintain FPS
    busy_wait(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))