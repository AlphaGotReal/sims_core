from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class SimConfig:
    frequency: int  = 200
    instances: int  = 1
    realtime:  bool = False
    render:    bool = False


@dataclass
class RobotConfig:
    urdf: str = ""


@dataclass
class ControllerConfig:
    name:      str
    type:      str
    joints:    list[str]
    frequency: int
    params:    dict = field(default_factory=dict)


@dataclass
class SensorConfig:
    name:   str
    type:   str
    parent: str


@dataclass
class SceneObjectConfig:
    name:              str
    type:              str
    static:            bool       = True
    urdf:              str | None = None
    pose_distribution: dict       = field(default_factory=dict)


@dataclass
class SimFactoryConfig:
    sim:         SimConfig
    robot:       RobotConfig
    controllers: list[ControllerConfig]
    sensors:     list[SensorConfig]


@dataclass
class SceneConfig:
    objects: list[SceneObjectConfig] = field(default_factory=list)
    lights:  list[dict]              = field(default_factory=list)


def _parse_sim(d: dict) -> SimConfig:
    return SimConfig(**d)


def _parse_robot(d: dict) -> RobotConfig:
    return RobotConfig(**d)


def _parse_controller(d: dict) -> ControllerConfig:
    return ControllerConfig(
        name      = d["name"],
        type      = d["type"],
        joints    = d["joints"],
        frequency = d["frequency"],
        params    = d.get("params", {}),
    )


def _parse_sensor(d: dict) -> SensorConfig:
    return SensorConfig(**d)


def _parse_scene_object(d: dict) -> SceneObjectConfig:
    return SceneObjectConfig(
        name              = d["name"],
        type              = d["type"],
        static            = d.get("static", True),
        urdf              = d.get("urdf"),
        pose_distribution = d.get("pose_distribution", {}),
    )


def load_config(path: str) -> SimFactoryConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    assert "sim"         in raw, "config missing 'sim' section"
    assert "robot"       in raw, "config missing 'robot' section"
    assert "controllers" in raw, "config missing 'controllers' section"
    assert "sensors"     in raw, "config missing 'sensors' section"

    return SimFactoryConfig(
        sim         = _parse_sim(raw["sim"]),
        robot       = _parse_robot(raw["robot"]),
        controllers = [_parse_controller(c) for c in raw["controllers"]],
        sensors     = [_parse_sensor(s)     for s in raw["sensors"]],
    )


def load_scene_config(path: str) -> SceneConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    return SceneConfig(
        objects = [_parse_scene_object(o) for o in raw.get("objects", [])],
        lights  = raw.get("lights", []),
    )
