'''Cartpole environment using PyBullet physics.

Classic cart-pole system implemented by Rich Sutton et al.
    * http://incompleteideas.net/sutton/book/code/pole.c

Also see:
    * github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    * github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/bullet/cartpole_bullet.py
    * https://github.com/utiasDSL/safe-control-gym/blob/main/safe_control_gym/envs/gym_control/cartpole.py
'''

import os
import copy

import time
import xml.etree.ElementTree as etxml

import casadi as cs
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from gym.utils import seeding
from utils.symbolic_system import FirstOrderModel
from utils import utils

# from controllers.lqr.lqr import LQR
from utils.enum_class import Task
from envs.base_env import BaseEnv
from functools import partial


# from safe_control_gym.math_and_models.symbolic_systems import SymbolicModel
# from safe_control_gym.math_and_models.normalization import normalize_angle


class CartPole(BaseEnv):
    NAME = 'cartpole'
    URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'cartpole_template.urdf')

    INIT_STATE_RAND_INFO = {
        'init_x': {
            'distrib': 'uniform',
            'low': -0.05,
            'high': 0.05
        },
        'init_x_dot': {
            'distrib': 'uniform',
            'low': -0.05,
            'high': 0.05
        },
        'init_theta': {
            'distrib': 'uniform',
            'low': -0.05,
            'high': 0.05
        },
        'init_theta_dot': {
            'distrib': 'uniform',
            'low': -0.05,
            'high': 0.05
        }
    }

    def __init__(self, init_state=None, gui=False, **kwargs):
        # todo: something useless now, but will be useful later
        self.state = None
        # todo: super class!!
        super().__init__(**kwargs)

        # create a PyBullet client connection
        self.PYB_CLIENT = -1
        self.GUI = gui
        if self.GUI:
            self.PYB_CLIENT = p.connect(p.GUI)
        else:
            self.PYB_CLIENT = p.connect(p.DIRECT)
        # disable urdf caching for randomization via reload urdf
        p.setPhysicsEngineParameter(enableFileCaching=1)

        # set gui and rendering size
        self.RENDER_HEIGHT = int(200)
        self.RENDER_WIDTH = int(320)

        # set the init state
        # (x, x_dot, theta, theta_dot)
        self.nState = 4
        self.nControl = 1
        if init_state is None:
            self.INIT_X, self.INIT_THETA, self.INIT_X_DOT, self.INIT_THETA_DOT = np.zeros(self.nState)
        elif isinstance(init_state, np.ndarray) and len(init_state) == self.nState:
            self.INIT_X, self.INIT_THETA, self.INIT_X_DOT, self.INIT_THETA_DOT = init_state
        else:
            raise ValueError('[ERROR] in CartPole.__init__(), init_state, type: {}, size: {}'.format(type(init_state),
                                                                                                     len(init_state)))

        # get physical properties from URDF (as default parameters)
        self.GRAVITY = 9.81
        self.EFFECTIVE_POLE_LENGTH, self.POLE_MASS, self.CART_MASS = self._parse_urdf_parameters(self.URDF_PATH)


        # self._setup_symbolic()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.nState,), dtype=np.float32)

    def reset(self, seed=None):
        # reset environment
        super().before_reset(seed=seed)
        p.resetSimulation(physicsClientId=self.PYB_CLIENT)
        p.setGravity(0, 0, -self.GRAVITY, physicsClientId=self.PYB_CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.PYB_CLIENT)
        p.setRealTimeSimulation(enableRealTimeSimulation=0, physicsClientId=self.PYB_CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.PYB_CLIENT)

        # load the cartpole urdf
        self.CARTPOLE_ID = p.loadURDF(self.URDF_PATH, basePosition=[0, 0, 0], physicsClientId=self.PYB_CLIENT)
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.PYB_CLIENT)

        # force control
        for i in [-1, 0, 1]:  # Slider, cart, and pole.
            p.changeDynamics(self.CARTPOLE_ID, linkIndex=i, linearDamping=0, angularDamping=0,
                             physicsClientId=self.PYB_CLIENT)
        for i in [0, 1]:  # Slider-to-cart and cart-to-pole joints.
            p.setJointMotorControl2(self.CARTPOLE_ID, jointIndex=i, controlMode=p.VELOCITY_CONTROL, force=0,
                                    physicsClientId=self.PYB_CLIENT)

        # set the initial state
        p.resetJointState(
            self.CARTPOLE_ID,
            jointIndex=0,  # Slider-to-cart joint.
            targetValue=self.INIT_X,
            targetVelocity=self.INIT_X_DOT,
            physicsClientId=self.PYB_CLIENT)
        p.resetJointState(
            self.CARTPOLE_ID,
            jointIndex=1,  # Cart-to-pole joints.
            targetValue=self.INIT_THETA,
            targetVelocity=self.INIT_THETA_DOT,
            physicsClientId=self.PYB_CLIENT)

        return self.get_state(), {}

    def step(self, action):
        for _ in range(self.PYB_STEPS_PER_CTRL):
            # apply force to cartpole
            p.setJointMotorControl2(self.CARTPOLE_ID,
                                    jointIndex=0,  # slider-to-cart joint
                                    controlMode=p.TORQUE_CONTROL,
                                    force=action,
                                    physicsClientId=self.PYB_CLIENT)
            p.stepSimulation(physicsClientId=self.PYB_CLIENT)

        self.state = self.get_state()

        reward = self._compute_reward(self.state, action)

        done = self._is_done(self.state)

        return self.state, reward, done, {}, {}

    def _compute_reward(self, state, action):
        x, theta, x_dot, theta_dot = state
        # reward = 1 - np.cos(theta)
        reward = 1 - np.cos(theta) - 1* np.square(x_dot) - 1 * np.square(theta_dot)
        return reward

    def _is_done(self, state):
        x, theta, x_dot, theta_dot = state
        if np.abs(x) > 2.4 or np.abs(theta) > 12 * 2 * np.pi / 360:
            return True
        else:
            return False

    @property
    def get_id(self):
        return self.id

    def get_state(self):
        # [x, theta, x_dot, theta_dot]
        x = p.getJointState(self.CARTPOLE_ID, jointIndex=0, physicsClientId=self.PYB_CLIENT)[0]
        x_dot = p.getJointState(self.CARTPOLE_ID, jointIndex=0, physicsClientId=self.PYB_CLIENT)[1]
        theta = p.getJointState(self.CARTPOLE_ID, jointIndex=1, physicsClientId=self.PYB_CLIENT)[0]
        theta_dot = p.getJointState(self.CARTPOLE_ID, jointIndex=1, physicsClientId=self.PYB_CLIENT)[1]
        theta = utils.normalize_angle(theta)
        state = np.hstack((x, theta, x_dot, theta_dot))

        self.state = np.array(state)
        return self.state


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _parse_urdf_parameters(self, file_name):
        '''Parses an URDF file for the robot's properties.

        Args:
            file_name (str, optional): The .urdf file from which the properties should be pased.

        Returns:
            EFFECTIVE_POLE_LENGTH (float): The effective pole length.
            POLE_MASS (float): The pole mass.
            CART_MASS (float): The cart mass.
        '''
        URDF_TREE = (etxml.parse(file_name)).getroot()
        EFFECTIVE_POLE_LENGTH = 0.5 * float(
            URDF_TREE[3][0][0][0].attrib['size'].split(' ')[-1])  # Note: HALF length of pole.
        POLE_MASS = float(URDF_TREE[3][1][1].attrib['value'])
        CART_MASS = float(URDF_TREE[1][2][0].attrib['value'])
        return EFFECTIVE_POLE_LENGTH, POLE_MASS, CART_MASS



    def _preprocess_control(self, action):
        action = self._denormalize_action(action)
        self.current_physical_action = action
        # todo: add action disturbances here
        self.current_noisy_physical_action = action
        # clip action
        force = np.clip(action, self.physical_action_bounds[0], self.physical_action_bounds[1])  # nonempty
        self.current_clipped_action = force

        return force[0]

    def _set_action_space(self):
        self.action_limit = 10
        self.physical_action_bounds = (-1 * np.atleast_1d(self.action_limit), -1 * np.atleast_1d(self.action_limit))
        self.action_threshold = 1
        self.action_space = spaces.Box(low=-self.action_threshold, high=self.action_threshold, shape=(1,))

        # define action/input labels and units
        self.ACTION_LABELS = ['U']
        self.ACTION_UNITS = ['N']

    def _denormalize_action(self, action):
        """ converts a normalized action into a physical action, only need in RL-based action
        :param action (ndarray):
        :return: action (ndarray):
        """
        return action


if __name__ == '__main__':
    key_word = {'gui': False}
    env_func = partial(CartPole, **key_word)
    # q_lqr = [1]
    # r_lqr = [0.1]
    # lqr_controller = LQR(env_func=env_func, q_lqr=q_lqr, r_lqr=r_lqr, discrete_dynamics=True)
    print("start")
    init_state = np.array([0, 0.3, 0, 0])
    cart_pole = CartPole(gui=True, init_state=init_state)
    cart_pole.reset()
    i = 0
    while 1:
        current_state = cart_pole.get_state()
        # if i < 800:
        #     action = np.array([20])
        #     cart_pole.step(action)
        # i += 1
        # print(cart_pole.get_state())
        time.sleep(0.1)
    print("cart pole dyn func: {}".format(cart_pole.symbolic.fc_func))
    while 1:
        pass
