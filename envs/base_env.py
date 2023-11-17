from abc import ABC, abstractmethod
import gymnasium as gym
import os
from gym.utils import seeding
import numpy as np

# self defined class
from utils.enum_class import Task



class BaseEnv(gym.Env, ABC):
    _count = 0
    NAME = 'base'

    def __init__(self,
                 gui: bool = False,
                 seed=None,
                 task: Task = Task.STABILIZATION,
                 randomized_init: bool = True,
                 pyb_freq: int = 50,
                 ctrl_freq: int = 50,
                 output_dir: str = None,
                 **kwargs):
        self.idx = self.__class__._count
        self.__class__._count += 1
        self.GUI = gui
        self.Task = task
        self.CTRL_FREQ = ctrl_freq  # control frequency
        self.PYB_FREQ = pyb_freq  # simulator frequency
        # simulator frequency should larger than control frequency and should be divisible by ctrl one
        if self.PYB_FREQ % self.CTRL_FREQ != 0:
            raise ValueError('[ERROR] in BaseEnv.__init__(), pyb_freq is not divisible by env_freq.')
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)
        self.CTRL_TIMESTEP = 1. / self.CTRL_FREQ
        self.PYB_TIMESTEP = 1. / self.PYB_FREQ
        self.RANDOMIZED_INIT = randomized_init
        if output_dir is None:
            output_dir = os.getcwd()
        self.output_dir = output_dir
        self.np_random, seed = seeding.np_random(seed)
        self.seed(seed)

        self.inited = False
        self.in_reset = False
        self.pyb_step_counter = 0
        self.ctrl_step_counter = 0
        self.current_raw_action = None
        self.current_physical_action = None  # current_raw_action unnormalized if it was normalized
        self.current_noisy_physical_action = None  # current_physical_action with noise added
        self.current_clipped_action = None  # current_noisy_physical_action clipped to physical action bounds

        self._set_action_space()

    def before_reset(self, seed=None):
        self.inited = True
        self.in_reset = True

    def before_step(self, action):
        """

        :param action (ndarray/scalar): The raw action returned by the controller.
        :return: action (ndarray): the processed action to be executed
        """
        if not self.inited:
            raise ValueError('[ERROR]: not init before call step().')

        action = np.atleast_1d(action)
        if action.ndim != 1 or action[0] is None:
            raise ValueError('[ERROR]: The action returned by the controller must be 1 dimensional.')

        self.current_raw_action = action

        processed_action = self._preprocess_control(action)

        return processed_action

    @abstractmethod
    def _preprocess_control(self, action):
        raise NotImplementedError

    @abstractmethod
    def _denormalize_action(self, action):
        """converts a normalized action into a physical action, only need in RL-based action
        :param action (ndarray):
        :return: action (ndarray):
        """
        raise NotImplementedError

    @abstractmethod
    def _set_action_space(self):
        """
        Defines the action space of the environment
        :return: None
        """
        raise NotImplementedError

    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        # todo: seed for action space and disturbs
        # self.action_space.seed(seed)
        # for _, disturbs in self.disturbances.items():
        #     disturbs.seed(self)
        return [seed]

