from __future__ import annotations

import numpy as np

from .base import BaseActor, register


@register
class ZeroActor(BaseActor):
    def update(self, obs: dict) -> None:
        print(self.world_idx)
        self.action[self.indices] = 0.0

