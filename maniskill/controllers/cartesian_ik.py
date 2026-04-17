from __future__ import annotations

import numpy as np
import torch
import sapien

from ..interfaces import InterfaceKey, StateInterface, CommandInterface
from .base        import Controller


def _angle_axis(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → angle-axis vector (rodrigues). Shape [3]."""
    cos_a = (np.trace(R) - 1.0) / 2.0
    cos_a = np.clip(cos_a, -1.0, 1.0)
    angle = np.arccos(cos_a)
    if abs(angle) < 1e-7:
        return np.zeros(3)
    axis  = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2.0 * np.sin(angle))
    return axis * angle


class CartesianIKController(Controller):
    """
    Differential IK: accepts end-effector pose targets, produces joint
    position commands each update.

    Uses damped-least-squares Jacobian inversion:
        dq = J^T (J J^T + λ² I)^{-1} v_ee
    where v_ee = [Δp; angle_axis(ΔR)] and λ is the Tikhonov damping.

    Requires a SAPIEN pinocchio model injected by Entity after robot init:
        controller.set_pinocchio_model(robot.create_pinocchio_model(), ee_link_idx)
    """

    def __init__(
        self,
        name:      str,
        joints:    list[str],
        ee_link:   str,
        frequency: int,
        gain:      float = 1.0,
        damping:   float = 1e-4,
    ) -> None:
        super().__init__(name, joints, frequency)
        self.ee_link      = ee_link
        self.gain         = gain
        self.damping      = damping
        self._target_pose: sapien.Pose | None = None
        self._pin_model   = None
        self._ee_link_idx: int | None = None

    @property
    def state_interface_keys(self) -> list[InterfaceKey]:
        return [f"{j}/position" for j in self.controlled_joints]

    @property
    def command_interface_keys(self) -> list[InterfaceKey]:
        return [f"{j}/position" for j in self.controlled_joints]

    def set_pinocchio_model(self, model, ee_link_idx: int) -> None:
        self._pin_model   = model
        self._ee_link_idx = ee_link_idx

    def reset(self) -> None:
        self._target_pose = None

    def set_target_pose(self, pose: sapien.Pose) -> None:
        self._target_pose = pose

    def update(
        self,
        state:    StateInterface,
        commands: CommandInterface,
    ) -> None:
        if self._target_pose is None or self._pin_model is None:
            return

        q_np = np.array([
            state[f"{j}/position"].squeeze().cpu().numpy()
            for j in self.controlled_joints
        ])  # [n_joints] — single env, IK is CPU-based

        self._pin_model.compute_forward_kinematics(q_np)
        ee_pose = self._pin_model.get_link_pose(self._ee_link_idx)

        p_err = (
            np.array(self._target_pose.p)
            - np.array(ee_pose.p)
        )
        R_cur    = sapien.Pose(q=ee_pose.q).to_transformation_matrix()[:3, :3]
        R_tgt    = sapien.Pose(q=self._target_pose.q).to_transformation_matrix()[:3, :3]
        r_err    = _angle_axis(R_tgt @ R_cur.T)
        v_ee     = np.concatenate([p_err, r_err])  # [6]

        J        = self._pin_model.compute_jacobian(q_np, [self._ee_link_idx])
        lam      = self.damping ** 2
        JJT      = J @ J.T + lam * np.eye(6)
        dq       = self.gain * J.T @ np.linalg.solve(JJT, v_ee)  # [n_joints]

        dt       = 1.0 / self.frequency
        q_new    = torch.tensor(q_np + dt * dq, dtype=torch.float32)

        for i, j in enumerate(self.controlled_joints):
            commands[f"{j}/position"] = q_new[i].unsqueeze(0)
