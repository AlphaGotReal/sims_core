from abc import ABC, abstractmethod

from ..interfaces import InterfaceKey, StateInterface, CommandInterface


class Controller(ABC):

    def __init__(
        self,
        name:      str,
        joints:    list[str],
        frequency: int,
    ) -> None:
        self.name              = name
        self.controlled_joints = joints
        self.frequency         = frequency

    @property
    @abstractmethod
    def state_interface_keys(self) -> list[InterfaceKey]: ...

    @property
    @abstractmethod
    def command_interface_keys(self) -> list[InterfaceKey]: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def update(
        self,
        state:    StateInterface,
        commands: CommandInterface,
    ) -> None: ...

    def should_update(self, sim_step: int, sim_freq: int) -> bool:
        stride = max(1, sim_freq // self.frequency)
        return sim_step % stride == 0
