from __future__ import annotations

import gymnasium as gym

from .config      import SimFactoryConfig, SceneConfig, load_config, load_scene_config
from .agent       import from_config as make_agent_class
from .env         import SimFactoryEnv   # registers SimFactory-v0
from .entity      import Entity, ControllerManager
from .controllers import (
    Controller,
    JointPositionController,
    JointVelocityController,
    JointTorqueController,
    CartesianIKController,
    RLController,
)


CONTROLLER_REGISTRY: dict[str, type[Controller]] = {
    "joint_position": JointPositionController,
    "joint_velocity": JointVelocityController,
    "joint_torque":   JointTorqueController,
    "cartesian_ik":   CartesianIKController,
    "rl":             RLController,
}


def _build_controller(cfg) -> Controller:
    cls = CONTROLLER_REGISTRY.get(cfg.type)
    assert cls is not None, (
        f"Unknown controller type '{cfg.type}'. "
        f"Available: {list(CONTROLLER_REGISTRY)}"
    )
    return cls(
        name      = cfg.name,
        joints    = cfg.joints,
        frequency = cfg.frequency,
        **cfg.params,
    )


def _collect_all_joints(factory_cfg: SimFactoryConfig) -> list[str]:
    seen   = {}
    joints = []
    for ctrl_cfg in factory_cfg.controllers:
        for j in ctrl_cfg.joints:
            if j not in seen:
                seen[j] = True
                joints.append(j)
    return joints


def _collect_ee_links(controllers: list[Controller]) -> list[str]:
    return [
        ctrl.ee_link
        for ctrl in controllers
        if isinstance(ctrl, CartesianIKController)
    ]


def build_entity(
    config_path:       str,
    scene_config_path: str | None = None,
    render_mode:       str        = "rgb_array",
    sim_backend:       str        = "auto",
    obs_mode:          str        = "sensor_data",
    agent_uid:         str        = "configurable_robot",
) -> Entity:
    """
    Full pipeline:
        config YAML
        → SimFactoryConfig
        → ConfigurableAgent (dynamically registered)
        → SimFactoryEnv (gym.make)
        → ControllerManager (validated)
        → Entity
    """
    factory_cfg  = load_config(config_path)
    scene_cfg    = load_scene_config(scene_config_path) if scene_config_path else None

    controllers  = [_build_controller(c) for c in factory_cfg.controllers]
    all_joints   = _collect_all_joints(factory_cfg)
    ee_links     = _collect_ee_links(controllers)

    make_agent_class(
        robot_cfg     = factory_cfg.robot,
        sensor_cfgs   = factory_cfg.sensors,
        all_joints    = all_joints,
        ee_link_names = ee_links,
        uid           = agent_uid,
    )

    env = gym.make(
        "SimFactory-v0",
        num_envs        = factory_cfg.sim.instances,
        obs_mode        = obs_mode,
        control_mode    = "pd_joint_pos",
        render_mode     = render_mode if factory_cfg.sim.render else None,
        sim_backend     = sim_backend,
        robot_uids      = agent_uid,
        factory_config  = factory_cfg,
        scene_config    = scene_cfg,
    )

    sensor_name_map = {s.name: s.name for s in factory_cfg.sensors}

    manager = ControllerManager(controllers)

    return Entity(
        env                = env,
        controller_manager = manager,
        config             = factory_cfg,
        sensor_name_map    = sensor_name_map,
    )
