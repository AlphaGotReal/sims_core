#! /usr/bin/env python

import argparse
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv

from _mj_utils import load_model_with_cameras, CameraStreams


class MyRobotEnv(gym.Env):

    def __init__(self, xml_path: str, render: bool = False):
        super().__init__()
        self.model, self.cams = load_model_with_cameras(xml_path)
        self.data             = mujoco.MjData(self.model)

        self.action_space = spaces.Box(-1.0, 1.0,
                                       shape=(self.model.nu,), dtype=np.float32)
        obs_size = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(obs_size,), dtype=np.float32)

        self._streams = (CameraStreams(self.model, self.cams, show=True)
                         if render else None)

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).astype(np.float32)

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        if self._streams is not None:
            self._streams.update(self.data)
        return self._get_obs(), 0.0, False, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def close(self):
        if self._streams is not None:
            self._streams.close()


def main(args):

    # cv2 + AsyncVectorEnv subprocesses don't mix; force single env on --render.
    if args.render:
        env = MyRobotEnv(args.xml, render=True)
        obs, _ = env.reset()
        print(f"obs shape : {obs.shape}")
        for _ in range(args.steps):
            env.step(env.action_space.sample())
        env.close()
        return

    envs = AsyncVectorEnv(
        [lambda: MyRobotEnv(args.xml) for _ in range(args.num_envs)]
    )
    obs, _ = envs.reset()
    print(f"batched obs shape : {obs.shape}")
    for _ in range(args.steps):
        envs.step(envs.action_space.sample())
    envs.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("xml")
    parser.add_argument("--render",   action="store_true",
                        help="show cv2 camera windows (forces single env)")
    parser.add_argument("--num-envs", type=int, default=3)
    parser.add_argument("--steps",    type=int, default=500)
    args = parser.parse_args()
    main(args)
