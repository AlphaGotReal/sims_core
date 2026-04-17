from __future__ import annotations

import os
from dataclasses import dataclass, field

import yaml


@dataclass
class MimicJoint:
    joint:      str           # controlling joint name
    multiplier: float = 1.0
    offset:     float = 0.0


@dataclass
class ActuatorConfig:
    type:             str                    # pd_joint_pos | pd_joint_pos_mimic
    joints:           list[str]              = field(default_factory=list)   # empty = all joints
    stiffness:        list[float] | float    = 1e3    # scalar or per-joint list
    damping:          list[float] | float    = 1e2
    force_limit:      list[float] | float    = 100.0
    normalize_action: bool                   = False
    mimic:            dict[str, MimicJoint]  = field(default_factory=dict)  # pd_joint_pos_mimic only


@dataclass
class CameraConfig:
    name:   str
    link:   str
    pose:   list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    width:  int         = 128
    height: int         = 128
    fov:    float       = 1.5708   # radians (~90°)
    near:   float       = 0.01
    far:    float       = 100.0


@dataclass
class HardwareConfig:
    actuators: list[ActuatorConfig] = field(default_factory=list)
    cameras:   list[CameraConfig]   = field(default_factory=list)


def load(path: str) -> HardwareConfig:
    base = os.path.dirname(os.path.abspath(path))

    with open(path) as f:
        raw = yaml.safe_load(f)

    def parse_gain(v, default):
        if v is None:
            return default
        return [float(x) for x in v] if isinstance(v, list) else float(v)

    actuators = [
        ActuatorConfig(
            type             = a["type"],
            joints           = a.get("joints", []),
            stiffness        = parse_gain(a.get("stiffness"),   1e3),
            damping          = parse_gain(a.get("damping"),     1e2),
            force_limit      = parse_gain(a.get("force_limit"), 100.0),
            normalize_action = bool(a.get("normalize_action", False)),
            mimic            = {
                name: MimicJoint(
                    joint      = m["joint"],
                    multiplier = float(m.get("multiplier", 1.0)),
                    offset     = float(m.get("offset",     0.0)),
                )
                for name, m in a.get("mimic", {}).items()
            },
        )
        for a in raw.get("actuators", [])
    ]

    cameras = [
        CameraConfig(
            name   = c["name"],
            link   = c["link"],
            pose   = c.get("pose",   [0.0, 0.0, 0.0]),
            width  = int(c.get("width",  128)),
            height = int(c.get("height", 128)),
            fov    = float(c.get("fov",  1.5708)),
            near   = float(c.get("near", 0.01)),
            far    = float(c.get("far",  100.0)),
        )
        for c in raw.get("cameras", [])
    ]

    return HardwareConfig(actuators=actuators, cameras=cameras)
