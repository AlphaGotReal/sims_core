from __future__ import annotations

from collections import deque
from typing import Protocol

import torch

from ..interfaces import InterfaceKey, StateInterface, CommandInterface
from .base        import Controller


class RLModel(Protocol):
    def predict(
        self,
        obs:          dict[str, torch.Tensor],
        prev_actions: torch.Tensor,
    ) -> torch.Tensor: ...


class RLController(Controller):
    """
    Policy-driven controller. Maintains a rolling observation window of
    depth `history_len` and calls model.predict() each update.

    observation_keys: which StateInterface keys to include in obs dict
    history_len:      how many past observations to stack per key
    """

    def __init__(
        self,
        name:             str,
        joints:           list[str],
        frequency:        int,
        model:            RLModel,
        observation_keys: list[InterfaceKey],
        history_len:      int = 1,
    ) -> None:
        super().__init__(name, joints, frequency)
        self.model            = model
        self.observation_keys = observation_keys
        self.history_len      = history_len
        self._obs_buffer: dict[str, deque] = {
            k: deque(maxlen=history_len) for k in observation_keys
        }
        self._prev_actions: torch.Tensor | None = None
        self._reward:       torch.Tensor | None = None

    @property
    def state_interface_keys(self) -> list[InterfaceKey]:
        return self.observation_keys

    @property
    def command_interface_keys(self) -> list[InterfaceKey]:
        return [f"{j}/position" for j in self.controlled_joints]

    def reset(self) -> None:
        for buf in self._obs_buffer.values():
            buf.clear()
        self._prev_actions = None
        self._reward       = None

    def set_reward(self, reward: torch.Tensor) -> None:
        self._reward = reward

    def update(
        self,
        state:    StateInterface,
        commands: CommandInterface,
    ) -> None:
        for k in self.observation_keys:
            if k in state:
                self._obs_buffer[k].append(state[k])

        all_filled = all(
            len(buf) == self.history_len
            for buf in self._obs_buffer.values()
        )
        if not all_filled:
            return

        obs = {
            k: torch.stack(list(buf), dim=1)
            for k, buf in self._obs_buffer.items()
        }

        n_joints  = len(self.controlled_joints)
        num_envs  = next(iter(obs.values())).shape[0]

        if self._prev_actions is None:
            self._prev_actions = torch.zeros(
                num_envs, n_joints,
                device = next(iter(obs.values())).device,
            )

        actions = self.model.predict(obs, self._prev_actions)
        self._prev_actions = actions.detach()

        for i, j in enumerate(self.controlled_joints):
            commands[f"{j}/position"] = actions[:, i]
