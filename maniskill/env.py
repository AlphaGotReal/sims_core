from __future__ import annotations

import numpy as np
import torch
import sapien

import gymnasium as gym
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.types import SimConfig

from .config import SimFactoryConfig, SceneConfig, SceneObjectConfig


def _sample_uniform(distribution: dict, device: torch.device) -> torch.Tensor:
    xyz_range = distribution.get("xyz_range", [[0, 0], [0, 0], [0, 0]])
    sample    = [
        np.random.uniform(lo, hi)
        for lo, hi in xyz_range
    ]
    return torch.tensor(sample, dtype=torch.float32, device=device)


@register_env("SimFactory-v0", max_episode_steps=100_000)
class SimFactoryEnv(BaseEnv):

    def __init__(
        self,
        *args,
        factory_config: SimFactoryConfig,
        scene_config:   SceneConfig | None = None,
        robot_uids:     str = "configurable_robot",
        **kwargs,
    ) -> None:
        self._factory_cfg = factory_config
        self._scene_cfg   = scene_config
        self._scene_actors: list = []
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye    = [0.6, 0.0, 0.6],
            target = [0.0, 0.0, 0.1],
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye    = [1.0, 1.0, 1.0],
            target = [0.0, 0.0, 0.3],
        )
        return CameraConfig("render_camera", pose, 1000, 1000, 1.0, 0.01, 100)

    def _load_scene(self, options: dict) -> None:
        if self._scene_cfg is None:
            return

        for obj in self._scene_cfg.objects:
            if obj.static and obj.urdf is not None:
                actor = self._load_actor(obj)
                self._scene_actors.append((obj, actor))

        self._randomize_lights()

    def _initialize_episode(
        self,
        env_idx:  torch.Tensor,
        options:  dict,
    ) -> None:
        if self._scene_cfg is None:
            return

        for obj, actor in self._scene_actors:
            if obj.static or not obj.pose_distribution:
                continue
            pos  = _sample_uniform(obj.pose_distribution, self.device)
            pose = sapien.Pose(p=pos.cpu().numpy())
            actor.set_pose(pose)

    def _load_actor(self, obj: SceneObjectConfig):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = obj.static
        actor  = loader.load(obj.urdf)
        actor.set_name(obj.name)
        return actor

    def _randomize_lights(self) -> None:
        if self._scene_cfg is None:
            return

        for light_cfg in self._scene_cfg.lights:
            intensity_range = light_cfg.get("intensity_range", [1000, 1000])
            pos_range       = light_cfg.get("position_range", [[0, 0], [0, 0], [2, 2]])
            intensity       = np.random.uniform(*intensity_range)
            pos             = np.array([np.random.uniform(*r) for r in pos_range])
            self.scene.add_point_light(pos, [intensity] * 3)

    def evaluate(self) -> dict:
        n = self.num_envs
        return {
            "success": torch.zeros(n, dtype=torch.bool, device=self.device),
            "fail":    torch.zeros(n, dtype=torch.bool, device=self.device),
        }

    def _get_obs_extra(self, info: dict) -> dict:
        return {}

    def compute_dense_reward(
        self, obs, action, info
    ) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(
        self, obs, action, info
    ) -> torch.Tensor:
        return self.compute_dense_reward(obs, action, info)
