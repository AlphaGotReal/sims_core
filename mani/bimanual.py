"""Bimanual dual-arm robot agent + a custom smoothed PD controller.

The URDF is at sims_core/robot.urdf and has:
    - 2 arms (left / right), each with 6 revolute joints + 2 prismatic (gripper)
    - 16 active joints in total
"""

from dataclasses import dataclass
from typing import Sequence, Union
from copy import deepcopy

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.registration import register_agent
from mani_skill.agents.controllers import (
    PDJointPosController,
    PDJointPosControllerConfig,
)
from mani_skill.sensors.camera import CameraConfig


URDF_PATH = "/home/straw/Projects/garage/sims_core/robot2.urdf"


# ---------------------------------------------------------------------------- #
# Custom controller: PD joint pos with exponential smoothing on the target.
#
# Why: raw delta-pos commands on a stiff manipulator can cause jerky motion
# and force spikes. We smooth the target with tau in [0, 1):
#     target_smooth_{t+1} = tau * target_smooth_{t} + (1 - tau) * target_raw
# This is a first-order low-pass filter over the commanded target.
# ---------------------------------------------------------------------------- #
class SmoothedPDJointPosController(PDJointPosController):
    config: "SmoothedPDJointPosControllerConfig"

    def reset(self):
        super().reset()
        self._smoothed_target = self._target_qpos.clone()

    def set_action(self, action):
        action        = self._preprocess_action(action)
        tau           = self.config.smoothing_tau
        self._step    = 0
        self._start_qpos = self.qpos

        if self.config.use_delta:
            raw_target = self._start_qpos + action
        else:
            raw_target = torch.broadcast_to(
                action, self._start_qpos.shape
            ).clone()

        self._smoothed_target = (
            tau * self._smoothed_target + (1.0 - tau) * raw_target
        )
        self._target_qpos = self._smoothed_target
        self.set_drive_targets(self._target_qpos)


@dataclass
class SmoothedPDJointPosControllerConfig(PDJointPosControllerConfig):
    smoothing_tau: float  = 0.8 # in [0, 1); higher = smoother/laggier
    controller_cls        = SmoothedPDJointPosController


# ---------------------------------------------------------------------------- #
# Agent
# ---------------------------------------------------------------------------- #
LEFT_ARM_JOINTS = [f"left_joint{i}"   for i in range(1, 7)]
RIGHT_ARM_JOINTS = [f"right_joint{i}" for i in range(1, 7)]
LEFT_GRIPPER_JOINTS  = ["left_joint7.1",  "left_joint7.2"]
RIGHT_GRIPPER_JOINTS = ["right_joint7.1", "right_joint7.2"]

ALL_ARM_JOINTS     = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
ALL_GRIPPER_JOINTS = LEFT_GRIPPER_JOINTS + RIGHT_GRIPPER_JOINTS


@register_agent()
class BimanualRobot(BaseAgent):

    uid            = "bimanual"
    urdf_path      = URDF_PATH
    fix_root_link  = True
    load_multiple_collisions = True

    arm_stiffness       = 1e3
    arm_damping         = 1e2
    arm_force_limit     = 100.0

    gripper_stiffness   = 1e3
    gripper_damping     = 1e2
    gripper_force_limit = 50.0

    keyframes = dict(
        rest=Keyframe(
            qpos = np.zeros(len(ALL_ARM_JOINTS) + len(ALL_GRIPPER_JOINTS)),
            pose = sapien.Pose(),
        ),
    )

    @property
    def _controller_configs(self):

        arm_pd_joint_pos = PDJointPosControllerConfig(
            ALL_ARM_JOINTS,
            lower         = None,
            upper         = None,
            stiffness     = self.arm_stiffness,
            damping       = self.arm_damping,
            force_limit   = self.arm_force_limit,
            normalize_action = False,
        )

        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            ALL_ARM_JOINTS,
            lower         = -0.1,
            upper         = 0.1,
            stiffness     = self.arm_stiffness,
            damping       = self.arm_damping,
            force_limit   = self.arm_force_limit,
            use_delta     = True,
        )

        gripper_pd_joint_pos = PDJointPosControllerConfig(
            ALL_GRIPPER_JOINTS,
            lower         = None,
            upper         = None,
            stiffness     = self.gripper_stiffness,
            damping       = self.gripper_damping,
            force_limit   = self.gripper_force_limit,
            normalize_action = False,
        )

        # Our custom controller — arm only, with smoothing.
        arm_smoothed_delta = SmoothedPDJointPosControllerConfig(
            ALL_ARM_JOINTS,
            lower         = -0.1,
            upper         = 0.1,
            stiffness     = self.arm_stiffness,
            damping       = self.arm_damping,
            force_limit   = self.arm_force_limit,
            use_delta     = True,
            smoothing_tau = 0.8,
        )

        return dict(
            pd_joint_pos = dict(
                arm     = arm_pd_joint_pos,
                gripper = gripper_pd_joint_pos,
            ),
            pd_joint_delta_pos = dict(
                arm     = arm_pd_joint_delta_pos,
                gripper = gripper_pd_joint_pos,
            ),
            pd_joint_smoothed_delta = dict(
                arm     = arm_smoothed_delta,
                gripper = gripper_pd_joint_pos,
            ),
        )

    def _after_init(self):
        # Save useful link handles for later (reward / obs).
        self.left_ee  = self.robot.find_link_by_name("left_end_effector")
        self.right_ee = self.robot.find_link_by_name("right_end_effector")

    # -------------------------------------------------------------------- #
    # Cameras
    #
    # Three RealSense cameras mounted via URDF-defined frames:
    #   - center_cam_link  : D455  (overhead, wider FOV)
    #   - left_cam_link    : D405  (wrist, stereo depth)
    #   - right_cam_link   : D405  (wrist, stereo depth)
    #
    # D405  : 848x480, h-FOV ~87 deg, working range 0.07 - 0.5 m
    # D455  : 848x480, h-FOV ~87 deg, working range 0.6 - 6.0 m
    #
    # ManiSkill takes a single `fov` param (vertical, radians). We derive it
    # from the horizontal FOV and the aspect ratio of the sensor.
    # -------------------------------------------------------------------- #
    @property
    def _sensor_configs(self):

        W, H = 848, 480

        # vertical FOV from horizontal FOV + aspect ratio
        def vfov(h_fov_rad):
            return 2.0 * np.arctan(np.tan(h_fov_rad / 2.0) * H / W)

        d405_fov = vfov(np.deg2rad(87.0))
        d455_fov = vfov(np.deg2rad(87.0))

        zero_pose = sapien.Pose()

        left_cam   = self.robot.find_link_by_name("left_cam_link")
        right_cam  = self.robot.find_link_by_name("right_cam_link")
        center_cam = self.robot.find_link_by_name("center_cam_link")

        return [
            CameraConfig(
                uid    = "left_wrist_d405",
                pose   = zero_pose,
                width  = W,  height = H,
                fov    = d405_fov,
                near   = 0.07, far = 0.5,
                mount  = left_cam,
            ),
            CameraConfig(
                uid    = "right_wrist_d405",
                pose   = zero_pose,
                width  = W,  height = H,
                fov    = d405_fov,
                near   = 0.07, far = 0.5,
                mount  = right_cam,
            ),
            CameraConfig(
                uid    = "center_d455",
                pose   = zero_pose,
                width  = W,  height = H,
                fov    = d455_fov,
                near   = 0.1,  far = 6.0,
                mount  = center_cam,
            ),
        ]
