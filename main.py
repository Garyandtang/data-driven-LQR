import control as ct
import numpy as np
import matplotlib.pyplot as plt

from decimal import *
getcontext().prec = 50

class data_pair:
    def __init__(self, n, m):
        self.x = np.zeros((n, 1), dtype=np.float64)
        self.u = np.zeros((m, 1), dtype=np.float64)
        self.x_next = np.zeros((n, 1), dtype=np.float64)
        self.K = np.zeros((m, n), dtype=np.float64)
        self.k = np.zeros((n, 1), dtype=np.float64)
        self.c = np.zeros((n, 1), dtype=np.float64)

# self.A = np.array([[0, 1], [-1, -1]])
#         self.B = np.array([[0], [1]])
#
#         self.Q = np.array([[1, 0], [0, 1]])
#         self.R = np.array([[1]])
#
#         self.K0 = np.array([[0.7, 0.5]])
class LTI:
    def __init__(self):
        self.A = np.array([[1.2, 0.5, 0.2], [0.1, 0.8, 0.3], [0.5, 0.1, 0.5]])
        self.B = np.array([[1, 0.1], [0.5, 0.2], [0.7, 1.0]])
        self.c = np.array([0, 0, 0])

        self.Q = np.array([[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]])
        self.R = np.array([[1, 0], [0, 1]])

        self.K0 = np.array([[-01.2098405, -0.47989766, -0.12446556],
                            [-0.47989766, -0.47989766, -0.92446556]])
        self.k0 = np.array([-1,-1])
        print("init K is stable: ", self.check_feedback_stabilizable(self.K0))

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
    print("K is stable: ", lti.check_feedback_stabilizable(K))
    data_vector = np.zeros(0, dtype=data_pair)
    x = np.random.uniform(-10, 10, (n,))
    for i in range(1000):
        # random generate u from uniform distribution [-3, 3]
        u = lti.action(x) + np.random.uniform(-20, 20, (m,))
        x_next = lti.step(x, u)
        data = data_pair(lti.B.shape[0], lti.B.shape[1])
        data.x = x
        data.u = u
        data.x_next = x_next
        data.K = K
        data.k = k
        data.c = lti.c
        data_vector = np.append(data_vector, data)
        x = x_next

    return data_vector

def learning():
    lti = LTI()
    K = lti.K0
    n = lti.A.shape[0]
    m = lti.B.shape[1]
    iteration = 19
    k_vector = np.zeros((n * m, iteration))
    k_vector = K.reshape((n * m, 1))[:, 0]
    for i in range(iteration-1):
        # random generate x0 from uniform distribution [-3, 3]
        data_vector = simulation(lti)
        K_prev = np.zeros((n*m,1))
        while np.linalg.norm(K.reshape((n*m,1))- K_prev) > 0.0001:

            K_prev = K.reshape((n*m,1))

            S = solve_S_from_data_collect(data_vector, lti.Q, lti.R)
            S_22 = S[n:n + m, n:n + m]
            S_12 = S[:n, n:n + m]
            K = -np.linalg.inv(S_22) @ S_12.T
            k_vector = np.vstack((k_vector, K.reshape((n * m, 1))[:, 0]))
            S_11 = S[:n, :n]
            B = lti.A @ np.linalg.inv(S_11 - lti.Q) @ S_12
            k = -np.linalg.pinv(B) @ lti.c
            lti.update_K0(K)
            lti.update_k0(k)
            print("k = ", k)
            print("error: ", np.linalg.norm(K.reshape((n * m, 1)) - K_prev))

    # plot K
    optimal_K = lti.get_optimal_K().reshape((n * m, 1))
    for i in range(k_vector.shape[1]):
        plt.figure()
        plt.plot(k_vector[:,i], label="K_{}".format(i))
        # plot a line
        plt.plot(np.ones((k_vector.shape[0], 1)) * optimal_K[i], label="K_{}_optimal".format(i))
        # set y limit
        plt.ylim(-1, 1)
        plt.legend()
        plt.show()


def B_Indentifier():
    lti = LTI()
    n = lti.A.shape[0]
    m = lti.B.shape[1]
    iteration = 10
    recovered_B_vector = np.zeros(0, dtype=np.ndarray)
    for i in range(iteration - 1):
        # random generate x0 from uniform distribution [-3, 3]
        data_vector = simulation(lti)
        # solve S from data
        S = solve_S_from_data_collect(data_vector, lti.Q, lti.R)
        # solve K from S
        S_22 = S[n:, n:]
        S_12 = S[:n, n:]
        K = -np.linalg.inv(S_22) @ S_12.T
        S_11 = S[:n, :n]
        B = lti.A @ np.linalg.inv(S_11 - lti.Q) @ S_12
        recovered_B_vector = np.append(recovered_B_vector, B)
        print("B = ", B)
        lti.update_K0(K)




def solve_S_from_data_collect(data_vector, Q, R):
    # S can be solved with least square method
    n = data_vector[0].x.shape[0]
    m = data_vector[0].u.shape[0]
    xi = np.zeros((n + m + n,))   # xi = [x; u; c]
    zeta = np.zeros((n + m + n,))  # zeta = [x_next; u_next; c]
    temp = np.kron(xi.T, zeta.T)
    A = np.zeros((data_vector.shape[0], temp.shape[0]))
    b = np.zeros((data_vector.shape[0],))
    for i in range(data_vector.shape[0]):
        x = data_vector[i].x
        u = data_vector[i].u
        xi[:n] = x
        xi[n:n+m] = u
        xi[n+m:] = data_vector[i].c
        zeta[:n] = data_vector[i].x_next
        u_next = data_vector[i].K @ data_vector[i].x_next + data_vector[i].k
        zeta[n:n+m] = u_next
        zeta[n+m:] = data_vector[i].c
        temp = np.kron(xi.T, xi.T) - np.kron(zeta.T, zeta.T)
        A[i, :] = temp
        b[i] = x.T @ Q @ x + u.T @ R @ u
    S = np.linalg.lstsq(A, b, rcond=None)[0]
    S = S.reshape((n + m + n, n + m + n))
    return S


if __name__ == '__main__':
    # B_Indentifier()
    learning()
    lti = LTI()
    print("optimal K = ", lti.get_optimal_K())




