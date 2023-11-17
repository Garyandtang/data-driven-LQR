import numpy as np
import casadi as ca
from utils.symbolic_system import FirstOrderModel
from liecasadi import SO3


# def _setup_symbolic(self, prior_prob={}):
#     l = prior_prob.get("pole_length", self.EFFECTIVE_POLE_LENGTH)
#     m = prior_prob.get("pole_mass", self.POLE_MASS)  # m2
#     M = prior_prob.get("cart_mass", self.CART_MASS)  # m1
#     Mm, ml = m + M, m * l
#     g = self.GRAVITY
#     dt = self.CTRL_TIMESTEP
#     # # Input variable
#     # q1 = cs.MX.sym('q1')  # x
#     # q2 = cs.MX.sym('q2')  # theta
#     # dq1 = cs.MX.sym('dq1')  # x_dot
#     # dq2 = cs.MX.sym('dq2')  # theta_dot
#     # q = cs.vertcat(q1, q2)
#     # dq = cs.vertcat(dq1, dq2)
#     # X = cs.vertcat(q, dq)
#     # U = cs.MX.sym('U')
#     # nx = self.nState
#     # nu = self.nControl
#     # # todo: it is different from safe-control-gym, check whether correct?
#     # ddq1 = (ml * cs.sin(q2) * dq2 ** 2 + U + m * g * cs.cos(q2) * cs.sin(q2)) / (M + m * (1 - cs.cos(q2) ** 2))
#     # ddq2 = -(ml * cs.cos(q2) * cs.sin(q2) * dq2 ** 2 + U * cs.cos(q2) + Mm * g * cs.sin(q2)) / (
#     #         l * M + ml * (1 - cs.cos(q2) ** 2))
#     # ddq = cs.vertcat(ddq1, ddq2)
#     # X_dot = cs.vertcat(dq, ddq)
#     # # observation
#     # Y = cs.vertcat(q, dq)
#
#     # todo: this is safe control gym
#     # Input variables.
#     x = cs.MX.sym('x')
#     x_dot = cs.MX.sym('x_dot')
#     theta = cs.MX.sym('theta')
#     theta_dot = cs.MX.sym('theta_dot')
#     X = cs.vertcat(x, theta, x_dot, theta_dot)
#     U = cs.MX.sym('U')
#     nx = 4
#     nu = 1
#     # Dynamics.
#     temp_factor = (U + ml * theta_dot ** 2 * cs.sin(theta)) / Mm
#     theta_dot_dot = (
#             (g * cs.sin(theta) - cs.cos(theta) * temp_factor) / (l * (4.0 / 3.0 - m * cs.cos(theta) ** 2 / Mm)))
#     X_dot = cs.vertcat(x_dot, theta_dot, temp_factor - ml * theta_dot_dot * cs.cos(theta) / Mm, theta_dot_dot)
#     # Observation.
#     Y = cs.vertcat(x, theta, x_dot, theta_dot)
#     # define cost (quadratic form)
#     Q = cs.MX.sym('Q', nx, nx)
#     R = cs.MX.sym('R', nu, nu)
#     Xr = cs.MX.sym('Xr', nx, 1)
#     Ur = cs.MX.sym('Ur', nu, 1)
#     cost_func = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (U - Ur).T * R * (U - Ur)
#     # define dyn and cost dictionaries
#     first_dynamics = {'dyn_eqn': X_dot, 'obs_eqn': Y, 'vars': {'X': X, 'U': U}}
#     cost = {'cost_func': cost_func, 'vars': {'X': X, 'Xr': Xr, 'U': U, 'Ur': Ur, 'Q': Q, 'R': R}}
#     # parameters
#     params = {
#         # prior inertial properties
#         'pole_length': l,
#         'pole_mass': m,
#         'cart_mass': M,
#         # equilibrium point for linearization
#         'X_EQ': np.array([1, 0, 0, 0]),
#         'U_EQ': np.zeros(self.nControl)  # np.atleast_2d(self.U_GOAL)[0, :],
#     }
#     self.symbolic = FirstOrderModel(dynamics=first_dynamics, dt=self.CTRL_TIMESTEP,
#                                     cost=cost, params=params)
