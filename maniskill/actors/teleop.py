from __future__ import annotations

import math
import threading
import tkinter as tk

from .base import BaseActor, register

PI = math.pi


@register
class TeleopActor(BaseActor):

    def __init__(self, cfg, world_idx, action, intermediate,
                 joint_names, sensor_names):
        super().__init__(cfg, world_idx, action, intermediate,
                         joint_names, sensor_names)
        lower = cfg.get("lower", [-PI] * len(self.indices))
        upper = cfg.get("upper", [ PI] * len(self.indices))
        t = threading.Thread(target=self._run_gui, args=(lower, upper), daemon=True)
        t.start()

    def _run_gui(self, lower, upper):
        root = tk.Tk()
        root.title(f"Teleop — world {self.world_idx}")
        root.resizable(False, False)

        for k, idx in enumerate(self.indices):
            name = self.joint_names[idx] if idx < len(self.joint_names) else f"joint_{idx}"
            lo   = lower[k] if k < len(lower) else -PI
            hi   = upper[k] if k < len(upper) else  PI

            row = tk.Frame(root)
            row.pack(fill=tk.X, padx=8, pady=2)

            tk.Label(row, text=name, width=22, anchor="w").pack(side=tk.LEFT)

            var = tk.DoubleVar(value=0.0)

            def on_slide(_, k=k, var=var):
                self.action[self.indices[k]] = var.get()

            tk.Scale(
                row, variable=var,
                from_=lo, to=hi,
                orient=tk.HORIZONTAL,
                resolution=0.001,
                length=320,
                command=on_slide,
            ).pack(side=tk.LEFT)

            tk.Label(row, textvariable=var, width=8).pack(side=tk.LEFT)

        root.mainloop()

    def update(self, obs: dict, t: float, dt: float) -> None:
        pass  # sliders write self.action directly via on_slide callbacks
