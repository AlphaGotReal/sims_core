from __future__ import annotations

from abc import abstractmethod

import numpy as np
import yaml


ACTOR_REGISTRY: dict[str, type] = {}


def register(cls):
    ACTOR_REGISTRY[cls.__name__] = cls
    return cls


class BaseActor:
    def __init__(self, cfg: dict, world_idx: int,
                 action: np.ndarray, intermediate: dict,
                 joint_names: list[str], sensor_names: list[str]):
        self.cfg          = cfg
        self.world_idx    = world_idx
        self.action       = action
        self.intermediate = intermediate
        self.joint_names  = joint_names
        self.sensor_names = sensor_names
        self.indices      = list(cfg["indices"])
        self.ikeys        = list(cfg.get("intermediate_writes", []))

    def update(self, obs: dict, t: float, dt: float) -> None:
        """Read obs, write self.action[self.indices]."""

    def step_callback(self, obs: dict, action: np.ndarray, new_obs: dict) -> bool:
        """Return True to trigger episode termination for this env."""
        return False

    def episode_callback(self) -> None:
        """Called when this env's episode ends, before reset."""

    def reset(self) -> None:
        """Reset actor internal state."""


def load(path: str, num_envs: int,
         actions: list[np.ndarray],
         intermediates: list[dict],
         joint_names: list[str],
         sensor_names: list[str]) -> list[list[BaseActor]]:
    with open(path) as f:
        entries = yaml.safe_load(f).get("actors", [])

    for entry in entries:
        assert entry["type"] in ACTOR_REGISTRY, \
            f"Unknown actor type '{entry['type']}'. Registered: {list(ACTOR_REGISTRY)}"

    all_actors = []
    for n in range(num_envs):
        seen_idx  = set()
        seen_keys = set()
        stack     = []
        for entry in entries:
            idxs  = set(entry["indices"])
            ikeys = set(entry.get("intermediate_writes", []))
            assert not (seen_idx & idxs), \
                f"Actor '{entry.get('name', entry['type'])}' has overlapping indices {seen_idx & idxs}"
            assert not (seen_keys & ikeys), \
                f"Actor '{entry.get('name', entry['type'])}' has overlapping intermediate keys {seen_keys & ikeys}"
            seen_idx.update(idxs)
            seen_keys.update(ikeys)
            cfg = {**entry, **entry.get("parameters", {})}
            stack.append(ACTOR_REGISTRY[entry["type"]](
                cfg, n, actions[n], intermediates[n], joint_names, sensor_names,
            ))  # n is world_idx
        all_actors.append(stack)
    return all_actors
