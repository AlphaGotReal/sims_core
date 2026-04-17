from __future__ import annotations

import torch

from .interfaces  import StateInterface, CommandInterface, InterfaceKey
from .controllers import Controller, CartesianIKController
from .config      import SimFactoryConfig


class ControllerManager:
    """
    Ordered sequence of controllers sharing StateInterface / CommandInterface.
    Validates at construction that no two controllers claim the same command key.
    """

    def __init__(self, controllers: list[Controller]) -> None:
        self._controllers = controllers
        self._validate_command_keys()

    def _validate_command_keys(self) -> None:
        seen: dict[InterfaceKey, str] = {}
        for ctrl in self._controllers:
            for key in ctrl.command_interface_keys:
                assert key not in seen, (
                    f"Command key '{key}' claimed by both "
                    f"'{seen[key]}' and '{ctrl.name}'"
                )
                seen[key] = ctrl.name

    def __getitem__(self, name: str) -> Controller:
        for ctrl in self._controllers:
            if ctrl.name == name:
                return ctrl
        raise KeyError(f"No controller named '{name}'")

    def reset(self) -> None:
        for ctrl in self._controllers:
            ctrl.reset()

    def update(
        self,
        state:    StateInterface,
        commands: CommandInterface,
        sim_step: int,
        sim_freq: int,
    ) -> None:
        for ctrl in self._controllers:
            if ctrl.should_update(sim_step, sim_freq):
                ctrl.update(state, commands)


class Entity:
    """
    Entangled pair: SimFactoryEnv + ControllerManager.
    Owns the read → update → write control loop.
    External code sets controller targets between step() calls.
    """

    def __init__(
        self,
        env:                gymnasium_env,
        controller_manager: ControllerManager,
        config:             SimFactoryConfig,
        sensor_name_map:    dict[str, str],  # sensor_name → camera_uid
    ) -> None:
        self._env                = env
        self.controller_manager  = controller_manager
        self._config             = config
        self._sensor_name_map    = sensor_name_map

        self._state:     StateInterface   = {}
        self._commands:  CommandInterface = {}
        self._sim_step:  int              = 0
        self._joint_order: list[str]      = []

    def reset(
        self,
        seed: int | None = None,
    ) -> tuple[dict, dict]:
        obs, info = self._env.reset(seed=seed)
        self.controller_manager.reset()
        self._sim_step = 0
        self._commands = {}
        self._build_joint_order()
        self._inject_pinocchio()
        self.read(obs)
        return obs, info

    def read(self, obs: dict) -> None:
        agent_obs = obs.get("agent", {})
        qpos      = agent_obs.get("qpos")
        qvel      = agent_obs.get("qvel")

        if qpos is not None and self._joint_order:
            for i, name in enumerate(self._joint_order):
                self._state[f"{name}/position"] = qpos[:, i]

        if qvel is not None and self._joint_order:
            for i, name in enumerate(self._joint_order):
                self._state[f"{name}/velocity"] = qvel[:, i]

        sensor_data = obs.get("sensor_data", {})
        for sensor_name, uid in self._sensor_name_map.items():
            cam = sensor_data.get(uid, sensor_data.get(sensor_name, {}))
            if "rgb" in cam:
                self._state[f"{sensor_name}/rgb"]   = cam["rgb"]
            if "depth" in cam:
                self._state[f"{sensor_name}/depth"] = cam["depth"]

        self._read_ee_poses()

    def _read_ee_poses(self) -> None:
        agent = self._env.unwrapped.agent
        for ctrl in self.controller_manager._controllers:
            if not isinstance(ctrl, CartesianIKController):
                continue
            link = agent.ee_links.get(ctrl.ee_link)
            if link is not None:
                self._state[f"{ctrl.ee_link}/pose"] = link.pose

    def update(self) -> None:
        self.controller_manager.update(
            self._state,
            self._commands,
            self._sim_step,
            self._config.sim.frequency,
        )

    def write(self) -> tuple[dict, float, bool, bool, dict]:
        action                           = self._build_action()
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._sim_step                  += 1
        self.read(obs)
        return obs, reward, terminated, truncated, info

    def step(self) -> tuple[dict, float, bool, bool, dict]:
        self.update()
        return self.write()

    def _build_action(self) -> torch.Tensor:
        n         = len(self._joint_order)
        num_envs  = self._env.unwrapped.num_envs

        # start from current positions so uncontrolled joints hold in place
        qpos = self._state.get(
            f"{self._joint_order[0]}/position"
            if self._joint_order else "",
            torch.zeros(num_envs),
        )
        device = qpos.device
        action = torch.zeros(num_envs, n, device=device)

        for i, name in enumerate(self._joint_order):
            pos_key = f"{name}/position"
            cmd_key = f"{name}/position"
            if cmd_key in self._commands:
                action[:, i] = self._commands[cmd_key]
            elif pos_key in self._state:
                action[:, i] = self._state[pos_key]

        return action

    def _build_joint_order(self) -> None:
        robot = self._env.unwrapped.agent.robot
        self._joint_order = [
            j.name for j in robot.get_active_joints()
        ]

    def _inject_pinocchio(self) -> None:
        """
        Injects pinocchio model + ee link indices into CartesianIKControllers.
        Called once after reset() when the robot actor is available.
        """
        robot = self._env.unwrapped.agent.robot
        pm    = robot.create_pinocchio_model()
        links = robot.get_links()
        link_name_to_idx = {l.name: i for i, l in enumerate(links)}

        for ctrl in self.controller_manager._controllers:
            if not isinstance(ctrl, CartesianIKController):
                continue
            idx = link_name_to_idx.get(ctrl.ee_link)
            assert idx is not None, (
                f"IK controller '{ctrl.name}': "
                f"ee_link '{ctrl.ee_link}' not found in robot"
            )
            ctrl.set_pinocchio_model(pm, idx)


# local alias avoids importing gymnasium at module level
gymnasium_env = object
