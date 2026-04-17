from __future__ import annotations

import torch

from ..interfaces import InterfaceKey, StateInterface, CommandInterface
from .base        import Controller


class JointPositionController(Controller):
    """
    Absolute or delta joint position control with optional exponential
    smoothing on the commanded target.

    Smoothing formula (first-order low-pass filter over commanded target):
        target_{t+1} = tau * target_t + (1 - tau) * target_raw
    Higher tau → smoother / laggier response.
    See bimanual.py:SmoothedPDJointPosController for the original.
    """

    def __init__(
        self,
        name:          str,
        joints:        list[str],
        frequency:     int,
        use_delta:     bool  = False,
        smoothing_tau: float = 0.0,
        lower:         float | None = None,
        upper:         float | None = None,
    ) -> None:
        super().__init__(name, joints, frequency)
        self.use_delta     = use_delta
        self.smoothing_tau = smoothing_tau
        self.lower         = lower
        self.upper         = upper
        self._target: torch.Tensor | None = None
        self._smoothed:  torch.Tensor | None = None

    @property
    def state_interface_keys(self) -> list[InterfaceKey]:
        return [f"{j}/position" for j in self.controlled_joints]

    @property
    def command_interface_keys(self) -> list[InterfaceKey]:
        return [f"{j}/position" for j in self.controlled_joints]

    def reset(self) -> None:
        self._target   = None
        self._smoothed = None

    def set_target(self, target: torch.Tensor) -> None:
        self._target = target

    def update(
        self,
        state:    StateInterface,
        commands: CommandInterface,
    ) -> None:
        if self._target is None:
            return

        q = torch.stack(
            [state[f"{j}/position"] for j in self.controlled_joints],
            dim = -1,
        )

        raw = (q + self._target) if self.use_delta else self._target

        if self.lower is not None:
            raw = raw.clamp(min=self.lower)
        if self.upper is not None:
            raw = raw.clamp(max=self.upper)

        if self._smoothed is None:
            self._smoothed = raw.clone()

        tau            = self.smoothing_tau
        self._smoothed = tau * self._smoothed + (1.0 - tau) * raw

        for i, j in enumerate(self.controlled_joints):
            commands[f"{j}/position"] = self._smoothed[..., i]
