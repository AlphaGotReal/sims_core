from __future__ import annotations

import torch

from ..interfaces import InterfaceKey, StateInterface, CommandInterface
from .base        import Controller


class JointTorqueController(Controller):
    """
    Impedance control: τ = K (q_d − q) + D (dq_d − dq)

    Writes effort commands. The corresponding joints in ConfigurableAgent
    must have ManiSkill PD gains zeroed so the sim does not double-count.

    NOTE: effort commands require direct SAPIEN force application via
    Entity.write(); they are NOT part of the flat pd_joint_pos action
    tensor. Entity handles this pathway separately.
    """

    def __init__(
        self,
        name:       str,
        joints:     list[str],
        frequency:  int,
        stiffness:  torch.Tensor,
        damping:    torch.Tensor,
    ) -> None:
        super().__init__(name, joints, frequency)
        self.stiffness    = stiffness
        self.damping      = damping
        self._q_desired:  torch.Tensor | None = None
        self._dq_desired: torch.Tensor | None = None

    @property
    def state_interface_keys(self) -> list[InterfaceKey]:
        pos = [f"{j}/position" for j in self.controlled_joints]
        vel = [f"{j}/velocity" for j in self.controlled_joints]
        return pos + vel

    @property
    def command_interface_keys(self) -> list[InterfaceKey]:
        return [f"{j}/effort" for j in self.controlled_joints]

    def reset(self) -> None:
        self._q_desired  = None
        self._dq_desired = None

    def set_desired(
        self,
        q_desired:  torch.Tensor,
        dq_desired: torch.Tensor,
    ) -> None:
        self._q_desired  = q_desired
        self._dq_desired = dq_desired

    def update(
        self,
        state:    StateInterface,
        commands: CommandInterface,
    ) -> None:
        if self._q_desired is None:
            return

        q  = torch.stack(
            [state[f"{j}/position"] for j in self.controlled_joints],
            dim = -1,
        )
        dq = torch.stack(
            [state[f"{j}/velocity"] for j in self.controlled_joints],
            dim = -1,
        )

        tau = (
            self.stiffness * (self._q_desired  - q)
            + self.damping * (self._dq_desired - dq)
        )

        for i, j in enumerate(self.controlled_joints):
            commands[f"{j}/effort"] = tau[..., i]
