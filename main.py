import control as ct
import numpy as np
import matplotlib.pyplot as plt
import scipy
from manifpy import SE2, SE2Tangent, SO2, SO2Tangent
from decimal import *
getcontext().prec = 100

class data_pair:
    def __init__(self, n, m):
        self.x = np.zeros((n,))
        self.u = np.zeros((m, ))
        self.x_next = np.zeros((n,))
class training_data:
    def __init__(self, n, m):
        self.data_container = data_pair(n, m)
        self.K = np.zeros((m, n))
        self.k = np.zeros((m,))
        self.c = np.zeros((n,))
        self.Q = np.zeros((n, n))
        self.R = np.zeros((m, m))
        self.A = np.zeros((n, n))
        self.B = np.zeros((n, m))

# self.A = np.array([[0, 1], [-1, -1]])
#         self.B = np.array([[0], [1]])
#
#         self.Q = np.array([[1, 0], [0, 1]])
#         self.R = np.array([[1]])
#
#         self.K0 = np.array([[0.7, 0.5]])
class LTI:
    def __init__(self):
        self.system_init()
        self.controller_init()

        self.arguemented_system_init()
        self.control_init_with_arguemented_system()


    def system_init(self):
        l = np.random.uniform(0.2, 1.3)
        r = np.random.uniform(0.03, 1.04)
        # l = 0.0518
        # r = 0.1908
        self.dt = 0.02
        self.v = 2
        self.w = 3
        twist = np.array([self.v, 0, self.w])
        adj = -SE2Tangent(twist).smallAdj()
        self.A = np.eye(3) + self.dt * adj
        self.B = self.dt * np.array([[r / 2, r / 2],
                                     [0, 0],
                                     [-r / l, r / l]])
        self.c = -self.dt * twist
        # self.c = np.zeros((3,))

        self.Q = 200 * np.eye(3)
        self.R = 0.2 * np.eye(2)

    def control_init_with_arguemented_system(self):
        self.K0_arg = ct.dlqr(0.9999*self.A_arg, self.B_arg, self.Q_arg, self.R_arg)[0]

    def arguemented_system_init(self):
        # argumented A, B
        n = self.A.shape[0]
        m = self.B.shape[1]
        A = np.zeros((n + n, n + n))
        A[:n, :n] = self.A
        A[:n, n:] = np.eye(n)
        A[n:, n:] = np.eye(n)
        B = np.zeros((n + n, m))
        B[:n, :] = self.B

        # set Q
        Q = np.zeros((n + n, n + n))
        Q[:n, :n] = self.Q

        # set R
        R = self.R

        self.A_arg = A
        self.B_arg = B
        self.Q_arg = Q
        self.R_arg = R


    def controller_init(self):
        # self.K0 = np.array([[-01.2098405, -0.47989766, -0.12446556],
        #                     [-0.47989766, -0.47989766, -0.92446556]])
        self.K0 = self.get_optimal_K()
        self.K0 = -ct.dlqr(self.A, 0.9*self.B, 0.1*self.Q, 10*self.R)[0]
        self.k0 = -np.linalg.pinv(0.1*self.B) @ self.c + np.random.uniform(2, 6, (2,))


    def step(self, x, u):
        # assert u.shape[0] == self.B.shape[1]
        # assert x.shape[0] == self.A.shape[0]
        x_next = self.A @ x + self.B @ u + self.c
        # check nan
        if np.isnan(x_next).any():
            print("nan")
        return x_next

    def action(self, x):
        return self.K0 @ x + self.k0

    def cost(self, x, u):
        return x.T @ self.Q @ x + u.T @ self.R @ u

    def get_optimal_K(self):
        K, S, E = ct.dlqr(self.A, self.B, self.Q, self.R)
        return -K

    def check_controllable(self):
        C = ct.ctrb(self.A, self.B)
        rank = np.linalg.matrix_rank(C)
        return rank == self.A.shape[0]

    def solve_with_dp(self):
        # argumented A, B
        n = self.A.shape[0]
        m = self.B.shape[1]
        A = np.zeros((n + n, n + n))
        A[:n, :n] = self.A
        A[:n, n:] = np.eye(n)
        A[n:, n:] = np.eye(n)
        B = np.zeros((n + n, m))
        B[:n, :] = self.B

        # set Q
        Q = np.zeros((n + n, n + n))
        Q[:n, :n] = self.Q

        # set R
        R = self.R

        # backward dp
        S = np.zeros((n + n, n + n))
        S_next = Q
        K = np.zeros((m, n + n))
        for i in range(100000):
            K = -np.linalg.inv(R + B.T @ S_next @ B) @ B.T @ S_next @ A
            S = A.T @ S_next @ A + Q + A.T @ S_next @ B @ K
            S_next = S
        print("K: ", K)

        K_12 = K[:, n:]
        print("res: ", self.B @ K_12)


    def check_feedback_stabilizable(self, K):
        assert K.shape[0] == self.B.shape[1] and K.shape[1] == self.A.shape[0]
        A = self.A + self.B @ K
        eig, _ = np.linalg.eig(A)
        res = True
        for e in eig:
            if np.linalg.norm(e) >= 1:
                res = False
                break
        return res

    def update_K0(self, K):
        self.K0 = K

    def update_k0(self, k):
        self.k0 = k

    def evaluation(self, K, k):
        x = np.array([3, 3, 2])
        nTraj = 3000
        x_container = np.zeros((3, nTraj))
        u_container = np.zeros((2, nTraj))
        for i in range(nTraj):
            u = K @ x + k
            x_next = self.step(x, u)
            x_container[:, i] = x
            u_container[:, i] = u
            x = x_next

        # plot
        plt.figure(1)
        plt.plot(x_container[0, :])
        plt.plot(x_container[1, :])
        plt.plot(x_container[2, :])
        plt.legend(['x', 'y', 'theta'])
        plt.show()

    def evaluate_arg_system(self, K):
        x = np.array([3, 3, 2])
        x_arg = np.append(x, self.c)
        nTraj = 3000
        x_container = np.zeros((6, nTraj))
        u_container = np.zeros((2, nTraj))
        for i in range(nTraj):
            u = K @ x_arg
            x_next = self.A_arg @ x_arg + self.B_arg @ u
            x_container[:, i] = x_arg
            u_container[:, i] = u
            x_arg = x_next
            print("x_arg: ", x_arg)

        # plot
        plt.figure(1)
        plt.plot(x_container[0, :])
        plt.plot(x_container[1, :])
        plt.plot(x_container[2, :])
        plt.legend(['x', 'y', 'theta'])
        plt.show()



def simulation(lti):
    K = lti.K0
    k = lti.k0
    print("init k: ", k)
    n = lti.A.shape[0]
    m = lti.B.shape[1]
    data = training_data(n, m)
    data.K = K
    data.k = k
    data.c = lti.c
    data.Q = lti.Q
    data.R = lti.R
    data.A = lti.A
    data.B = lti.B
    x = -0.1 + 0.20 * np.random.rand(n, )
    for i in range(150):
        u = K @ x + k -0.1 + 0.20 * np.random.rand(m, )
        x_next = lti.A @ x + lti.B @ u + lti.c
        pair = data_pair(n, m)
        pair.x = x
        pair.u = u
        pair.x_next = x_next
        if i == 0:
            data.data_container = pair
        else:
            data.data_container = np.append(data.data_container, pair)
        x = x_next

    return data


def learning(data_container, lti):
    n = data_container.K.shape[1]
    m = data_container.K.shape[0]
    iteration = 0
    recovered_B_vector = np.zeros(0, dtype=np.ndarray)
    K_prev = np.zeros((m, n))
    k_prev = np.zeros((m,))
    while np.linalg.norm(data_container.K - K_prev) > 0.01 or np.linalg.norm(data_container.k - k_prev) > 0.01:
        K_prev = data_container.K
        k_prev = data_container.k
        print("iteration: ", iteration, '===================')
        # solve S from data
        gamma = 1
        S = solve_S_from_data_collect(data_container, data_container.Q, data_container.R, gamma)
        # solve K from S
        S_22 = S[n:n + m, n:n + m]
        S_12 = S[:n, n:n + m]
        K = -np.linalg.inv(S_22) @ S_12.T
        S_11 = S[:n, :n]
        B = lti.A @ np.linalg.inv(S_11 - lti.Q) @ S_12
        recovered_B_vector = np.append(recovered_B_vector, B)
        print("B = ", B)
        k = -np.linalg.pinv(B) @ lti.c
        data_container.K = K
        data_container.k = k
        print("current K: ", K)
        print("current k: ", k)
        print("optimal K: ", lti.get_optimal_K())
        print("error K: ", data_container.K - K_prev)
        print("error k: ", data_container.k - k_prev)
        print("====================================")
        iteration += 1

    return K, k, B


def simulation_with_arguemented_system(lti):
    K = lti.K0_arg
    A = lti.A_arg
    B = lti.B_arg
    Q = lti.Q_arg
    R = lti.R_arg
    n = A.shape[0]
    m = B.shape[1]
    data = training_data(n, m)
    data.K = K
    data.k = np.zeros((m,))
    data.c = np.zeros((n,))
    data.Q = Q
    data.R = R
    data.A = A
    data.B = B
    x = np.random.normal(0, 3, (3,))
    x = np.append(x, lti.c)
    for i in range(300):
        # random generate u from uniform distribution [-3, 3]
        u = np.random.normal(0, 3, (2,))
        x_next = A @ x + B @ u
        pair = data_pair(n, m)
        pair.x = x
        pair.u = u
        pair.x_next = x_next
        if i == 0:
            data.data_container = pair
        else:
            data.data_container = np.append(data.data_container, pair)
        x = x_next
    return data






def solve_S_from_data_collect(data, Q, R, gamma):
    # S can be solved with least square method
    # gamma: discount factor
    n = data.K.shape[1]
    m = data.K.shape[0]
    K = data.K
    k = data.k
    c = data.c
    xi = np.zeros((n + m + n,))   # xi = [x; u; c]
    zeta = np.zeros((n + m + n,))  # zeta = [x_next; u_next; c]
    temp = np.kron(xi.T, zeta.T)
    A = np.zeros((data.data_container.shape[0], temp.shape[0]))
    b = np.zeros((data.data_container.shape[0],))
    for i in range(data.data_container.shape[0]):
        x = data.data_container[i].x
        u = data.data_container[i].u - k
        xi[:n] = x
        xi[n:n+m] = u
        xi[n+m:] = c
        zeta[:n] = data.data_container[i].x_next
        u_next = K @ data.data_container[i].x_next
        zeta[n:n+m] = u_next
        zeta[n+m:] = c
        temp = np.kron(xi.T, xi.T) - np.kron(zeta.T, zeta.T)
        A[i, :] = temp
        b[i] = np.power(gamma, i) * (x.T @ Q @ x + u.T @ R @ u)
    S = np.linalg.pinv(A) @ b
    S = S.reshape((2*n + m, 2*n + m))
    return S


if __name__ == '__main__':
    lti = LTI()
    lti.evaluation(lti.K0, lti.k0)
    data = simulation(lti)
    K, k, B = learning(data, lti)
    lti.evaluation(K, k)
    # learning()
    print("optimal B: ", lti.B)
    print("B error", lti.B - B)
    print("optimal K = ", lti.get_optimal_K())
    print("optimal k = ", -np.linalg.pinv(lti.B) @ lti.c)




