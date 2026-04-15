#! /usr/bin/env python
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv
import mujoco

# 1. You wrap your MuJoCo code in a standard Gym Environment
class MyRobotEnv(gym.Env):
    def __init__(self):
        super().__init__() # Initialize the Gymnasium parent class
        
        # Load your MuJoCo model from the command line argument
        xml_path = sys.argv[1] if len(sys.argv) > 1 else "your_model.xml"
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # ---------------------------------------------------------
        # DEFINE THE ACTION SPACE
        # ---------------------------------------------------------
        num_actions = self.model.nu
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(num_actions,), 
            dtype=np.float32
        )

        # ---------------------------------------------------------
        # DEFINE THE OBSERVATION SPACE
        # ---------------------------------------------------------
        # Using qpos (positions) and qvel (velocities) as the default observation
        obs_size = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )

    def _get_obs(self):
        # Helper function to extract and concatenate positions and velocities
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def step(self, action):
        # Apply the action to the motors (scale this if your XML expects > 1.0)
        self.data.ctrl[:] = action
        
        # Step the physics engine
        mujoco.mj_step(self.model, self.data)
        
        # Gather the required Gym returns
        obs = self._get_obs()
        reward = 0.0       # TODO: Define your reward logic here
        terminated = False # TODO: Define success/failure states here
        truncated = False  # TODO: Define episode timeout logic here
        info = {}
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Standard Gym procedure to seed the environment
        super().reset(seed=seed)
        
        # Reset MuJoCo state to default
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data) # Compute initial kinematics
        
        obs = self._get_obs()
        info = {}
        
        return obs, info

# 2. You use AsyncVectorEnv to spawn multiple processes
if __name__ == "__main__":
    # Ensure an XML path was provided via command line
    if len(sys.argv) < 2:
        print("Usage: python mujoco_gym.py <path_to_xml>")
        sys.exit(1)

    num_envs = 3 # Number of parallel environments
    
    print(f"Spawning {num_envs} parallel environments...")
    
    # This creates independent Python processes
    envs = AsyncVectorEnv([lambda: MyRobotEnv() for _ in range(num_envs)])
    
    # When you call reset or step, it executes across all cores simultaneously!
    observations, infos = envs.reset()
    print(f"Batched observation shape: {observations.shape}")
    
    # 'actions' is now a batch of actions. 
    # 'observations' is a batch of states.
    actions = envs.action_space.sample() 
    obs, rewards, terms, truncs, infos = envs.step(actions)
    
    print(f"Batched actions shape: {actions.shape}")
    print(f"Post-step batched observation shape: {obs.shape}")
    
    # Always close the environments when done to prevent zombie processes
    envs.close()
    print("Done!")
