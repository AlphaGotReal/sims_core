from __future__ import annotations

import numpy as np

from .base import BaseActor, register
import cv2

@register
class ZeroActor(BaseActor):
    def update(self, obs: dict, t: float, dt: float) -> None:
        self.action[self.indices] = 0.0

