import time

import control as ct
import numpy as np
import matplotlib.pyplot as plt
from envs.cartpole.cartpole import CartPole
class data_pair:
    def __init__(self, n, m):
        self.x = np.zeros((n, 1))
        self.u = np.zeros((m, 1))
        self.x_next = np.zeros((n, 1))
        self.K = np.zeros((m, n))

# -2.78974875,  -5.24884431, -44.98615057, -11.86937562
class fb_controller:
    def __init__(self, K = np.array([[ 44.98615057, 11.86937562,5.24884431]])):
        self.K = K

    def fb_control(self, state):
        u = self.K @ state
        return u

    def update_K(self, K):
        self.K = K

    def get_K(self):
        return self.K

def simulation(controller):
    K = controller.get_K()
    data_contrainer = np.zeros(0, dtype=data_pair)
    robot = CartPole()
    n = robot.n_lqr
    m = robot.m_lqr
    for i in range(100):
        x = robot.get_lqr_state()
        u = controller.fb_control(x) + np.random.normal(5, 5, (m,))
        x_next, _, _, _, _ = robot.step_lqr(u)
        # print("x_next: ", x_next)
        data = data_pair(n, m)
        data.x = x.reshape((n,))
        data.u = u
        data.x_next = x_next.reshape((n,))
        data.K = K
        data_contrainer = np.append(data_contrainer, data)

    return data_contrainer


def solve_S_from_data_collect(data_vector, Q, R, K0):
    # S can be solved with least square method
    n = data_vector[0].x.shape[0]
    m = data_vector[0].u.shape[0]
    xi = np.zeros((n + m,))   # xi = [x; u]
    zeta = np.zeros((n + m,))  # zeta = [x_next; u_next]
    temp = np.kron(xi.T, zeta.T)
    A = np.zeros((data_vector.shape[0], temp.shape[0]))
    b = np.zeros((data_vector.shape[0],))
    for i in range(data_vector.shape[0]):
        x = data_vector[i].x
        u = data_vector[i].u
        xi[:n] = x
        xi[n:] = u
        zeta[:n] = data_vector[i].x_next
        u_next = K0 @ data_vector[i].x_next
        zeta[n:] = u_next
        temp = np.kron(xi.T, xi.T) - np.kron(zeta.T, zeta.T)
        A[i, :] = temp
        b[i] = x.T @ Q @ x + u.T @ R @ u
    S = np.linalg.lstsq(A, b, rcond=None)[0]
    S = S.reshape((n + m, n + m))
    return S
def learning():
    iteration = 10
    robot = CartPole()
    controller = fb_controller()
    K = controller.get_K()
    print("init K: ", K)
    n = robot.n_lqr
    m = robot.m_lqr
    Q = robot.Q_lqr
    R = robot.R_lqr
    K_contrainer = np.zeros((n * m, iteration))
    K_contrainer[:, 0] = K.reshape((n * m))
    error_container = np.zeros((iteration, 1))
    for i in range(iteration - 1):
        data_container = simulation(controller)
        K_prev = np.zeros((n * m, 1))
        while np.linalg.norm(K - K_prev) > 0.01:
            print("error: ", np.linalg.norm(K - K_prev))
            K_prev = K

            S = solve_S_from_data_collect(data_container, Q, R, K)

            # solve K from S
            S_22 = S[n:, n:]
            S_12 = S[:n, n:]
            K = -np.linalg.inv(S_22) @ S_12.T
            controller.update_K(K)
            print("K = ", K)
        K_contrainer[:, i + 1] = K.reshape((n * m, 1))[:, 0]



    return K


if __name__ == '__main__':
    K = learning()
    K = np.array([[ 44.98615057, 11.86937562,5.24884431]])
    # K = np.array([[36.1250999,   9.31310334,  2.80320203]])
    controller = fb_controller(K)
    robot = CartPole(gui=True)
    while 1:
        state = robot.get_lqr_state()
        print("state: ", state)
        u = controller.fb_control(state) + np.random.uniform(-160, 160, (1,))
        robot.step_lqr(u)
        # print(u)
        time.sleep(0.01)