from __future__ import annotations

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent       import BaseAgent, Keyframe
from mani_skill.agents.registration     import register_agent
from mani_skill.agents.controllers      import (
    PDJointPosController,
    PDJointPosControllerConfig,
)
from mani_skill.sensors.camera          import CameraConfig

from .config     import RobotConfig, SensorConfig
from .controllers import CartesianIKController


CAMERA_NEAR = {"camera": 0.01}
CAMERA_FAR  = {"camera": 100.0}

CAMERA_NEAR_OVERRIDES = {
    "d405": 0.07,
    "d455": 0.1,
}
CAMERA_FAR_OVERRIDES = {
    "d405": 0.5,
    "d455": 6.0,
}

CAMERA_H_FOV_DEG = 87.0

ARM_STIFFNESS       = 1e3
ARM_DAMPING         = 1e2
ARM_FORCE_LIMIT     = 100.0
GRIPPER_STIFFNESS   = 1e3
GRIPPER_DAMPING     = 1e2
GRIPPER_FORCE_LIMIT = 50.0


def _vfov(h_fov_deg: float, width: int, height: int) -> float:
    h_fov_rad = np.deg2rad(h_fov_deg)
    return 2.0 * np.arctan(np.tan(h_fov_rad / 2.0) * height / width)


def _camera_near_far(sensor_name: str) -> tuple[float, float]:
    name_lower = sensor_name.lower()
    for key, near in CAMERA_NEAR_OVERRIDES.items():
        if key in name_lower:
            return near, CAMERA_FAR_OVERRIDES[key]
    return CAMERA_NEAR["camera"], CAMERA_FAR["camera"]


def from_config(
    robot_cfg:      RobotConfig,
    sensor_cfgs:    list[SensorConfig],
    all_joints:     list[str],
    ee_link_names:  list[str],
    uid:            str = "configurable_robot",
) -> type[BaseAgent]:
    """
    Dynamically creates and registers a BaseAgent subclass from config.
    The returned class can be passed to gym.make as robot_uids=uid.

    ManiSkill only sees a single pd_joint_pos mode covering all joints,
    keeping the action tensor layout deterministic. All higher-level
    control logic lives in the Controller layer above.
    """

    W, H       = 848, 480
    fov        = _vfov(CAMERA_H_FOV_DEG, W, H)
    zero_pose  = sapien.Pose()
    n_joints   = len(all_joints)

    def _controller_configs(self):
        return dict(
            pd_joint_pos = dict(
                all_joints = PDJointPosControllerConfig(
                    all_joints,
                    lower            = None,
                    upper            = None,
                    stiffness        = ARM_STIFFNESS,
                    damping          = ARM_DAMPING,
                    force_limit      = ARM_FORCE_LIMIT,
                    normalize_action = False,
                ),
            ),
        )

    def _sensor_configs(self):
        configs = []
        for s in sensor_cfgs:
            link        = self.robot.find_link_by_name(s.parent)
            near, far   = _camera_near_far(s.name)
            configs.append(CameraConfig(
                uid    = s.name,
                pose   = zero_pose,
                width  = W,
                height = H,
                fov    = fov,
                near   = near,
                far    = far,
                mount  = link,
            ))
        return configs

    def _after_init(self):
        self.ee_links = {
            name: self.robot.find_link_by_name(name)
            for name in ee_link_names
        }

    attrs = dict(
        uid                     = uid,
        urdf_path               = robot_cfg.urdf,
        fix_root_link           = True,
        load_multiple_collisions = True,
        all_joints              = all_joints,
        keyframes               = dict(
            rest = Keyframe(
                qpos = np.zeros(n_joints),
                pose = sapien.Pose(),
            ),
        ),
        _controller_configs     = property(_controller_configs),
        _sensor_configs         = property(_sensor_configs),
        _after_init             = _after_init,
    )

    cls = type(uid, (BaseAgent,), attrs)
    register_agent()(cls)
    return cls
