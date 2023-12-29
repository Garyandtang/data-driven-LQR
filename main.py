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
        self.k = np.zeros((n,))
        self.c = np.zeros((n,))

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


    def system_init(self):
        l = 0.0518
        r = 0.1908
        self.dt = 0.02
        self.v = 1
        self.w = 1
        twist = np.array([self.v, 0, self.w])
        adj = -SE2Tangent(twist).smallAdj()
        self.A = np.eye(3) + self.dt * adj
        self.B = self.dt * np.array([[r / 2, r / 2],
                                     [0, 0],
                                     [-r / l, r / l]])
        self.c = -self.dt * twist
        # self.c = np.zeros((3,))

        self.Q = 2 * np.eye(3)
        self.R = 2 * np.eye(2)


    def controller_init(self):
        # self.K0 = np.array([[-01.2098405, -0.47989766, -0.12446556],
        #                     [-0.47989766, -0.47989766, -0.92446556]])
        self.K0 = self.get_optimal_K()
        self.k0 = -np.linalg.pinv(self.B) @ self.c


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
    for i in range(50):
        # random generate u from uniform distribution [-3, 3]
        x = -100 + 200 * np.random.rand(n,)
        u =  -100 + 200 * np.random.rand(m,)
        x_next = lti.A @ x + lti.B @ u + lti.c
        pair = data_pair(n, m)
        pair.x = x
        pair.u = u
        pair.x_next = x_next
        if i == 0:
            data.data_container = pair
        else:
            data.data_container = np.append(data.data_container, pair)

    return data


def B_Indentifier():
    lti = LTI()
    n = lti.A.shape[0]
    m = lti.B.shape[1]
    iteration = 20
    recovered_B_vector = np.zeros(0, dtype=np.ndarray)
    data_container = simulation(lti)
    for i in range(iteration - 1):
        print("iteration: ", i, '===================')
        # solve S from data
        S = solve_S_from_data_collect(data_container, lti.Q, lti.R)
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
        print("====================================")
    print("K: ", K)




def solve_S_from_data_collect(data, Q, R):
    # S can be solved with least square method
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
        b[i] = x.T @ Q @ x + u.T @ R @ u
    S = np.linalg.pinv(A) @ b
    S = S.reshape((2*n + m, 2*n + m))
    return S


if __name__ == '__main__':
    B_Indentifier()
    # learning()
    lti = LTI()
    print("optimal B: ", lti.B)
    print("optimal K = ", lti.get_optimal_K())




