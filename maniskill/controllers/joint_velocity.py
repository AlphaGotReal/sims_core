from __future__ import annotations

import torch

from ..interfaces import InterfaceKey, StateInterface, CommandInterface
from .base        import Controller


class JointVelocityController(Controller):
    """
    Velocity control via numerical integration: q_target += vel * dt,
    where dt = 1 / frequency (controller period, not sim period).
    """

    def __init__(
        self,
        name:      str,
        joints:    list[str],
        frequency: int,
        gain:      float = 1.0,
    ) -> None:
        super().__init__(name, joints, frequency)
        self.gain      = gain
        self._velocity: torch.Tensor | None = None
        self._q_target: torch.Tensor | None = None

    @property
    def state_interface_keys(self) -> list[InterfaceKey]:
        pos = [f"{j}/position" for j in self.controlled_joints]
        vel = [f"{j}/velocity" for j in self.controlled_joints]
        return pos + vel

    @property
    def command_interface_keys(self) -> list[InterfaceKey]:
        return [f"{j}/position" for j in self.controlled_joints]

    def reset(self) -> None:
        self._velocity = None
        self._q_target = None

    def set_velocity(self, vel: torch.Tensor) -> None:
        self._velocity = vel

    def update(
        self,
        state:    StateInterface,
        commands: CommandInterface,
    ) -> None:
        if self._velocity is None:
            return

        q = torch.stack(
            [state[f"{j}/position"] for j in self.controlled_joints],
            dim = -1,
        )

        if self._q_target is None:
            self._q_target = q.clone()

        dt             = 1.0 / self.frequency
        self._q_target = self._q_target + self.gain * self._velocity * dt

        for i, j in enumerate(self.controlled_joints):
            commands[f"{j}/position"] = self._q_target[..., i]
