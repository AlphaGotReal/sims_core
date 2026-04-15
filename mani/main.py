#! /usr/bin/env python3

import argparse
import numpy as np
import torch
import gymnasium as gym
import sapien

import cv2
from IPython import embed

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.types import SimConfig

import bimanual  # registers "bimanual" agent and custom controller


@register_env("Empty-v0", max_episode_steps=20000)
class EmptyEnv(BaseEnv):

    SUPPORTED_ROBOTS = ["panda", "bimanual"]

    def __init__(self, *args, robot_uids="bimanual", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0.0, 0.6], target=[0.0, 0.0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[1.0, 1.0, 1.0], target=[0.0, 0.0, 0.3])
        return CameraConfig("render_camera", pose, 1000, 1000, 1.0, 0.01, 100)

    def _load_scene(self, options):
        return

    def _initialize_episode(self, env_idx, options):
        return

    def evaluate(self):
        n = self.num_envs
        return {
            "success": torch.zeros(n, dtype=torch.bool, device=self.device),
            "fail"   : torch.zeros(n, dtype=torch.bool, device=self.device),
        }

    def _get_obs_extra(self, info):
        return dict()

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)


def main(args):

    env = gym.make(
        "Empty-v0",
        num_envs     = 1,
        obs_mode     = args.obs_mode,
        control_mode = args.control_mode,
        render_mode  = args.render_mode,
        sim_backend  = args.sim_backend,
    )

    obs, info = env.reset(seed=0)
    print(f"obs space : {env.observation_space}")
    print(f"act space : {env.action_space}")

    if args.render_mode == "human":
        env.render()

    for step in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        for cam_n, cam_data in obs["sensor_data"].items():
            rgb   = cam_data["rgb"][0].cpu().numpy()
            depth = cam_data["depth"][0].cpu().numpy()
            cv2.waitKey(1)
            cv2.imshow(f"{cam_n}_rgb", rgb)
            cv2.imshow(f"{cam_n}_depth", depth)

        env.render()

    env.close()
    return env


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cont", action="store_true")
    parser.add_argument("--gvar", default="gvar")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--render-mode", default="human",
                        choices=["human", "rgb_array", "sensors"])
    parser.add_argument("--sim-backend", default="auto")
    parser.add_argument("--obs-mode", default="rgb+depth",
                        choices=["state", "rgb", "depth", "rgb+depth",
                                 "sensor_data", "pointcloud"])
    parser.add_argument("--control-mode", default="pd_joint_smoothed_delta",
                        choices=["pd_joint_pos", "pd_joint_delta_pos",
                                 "pd_joint_smoothed_delta"])
    args = parser.parse_args()

    _gvar = args.gvar
    globals()[_gvar] = main(args)

    if args.cont:
        embed()
