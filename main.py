import control as ct
import numpy as np
import matplotlib.pyplot as plt

class data_pair:
    def __init__(self, n, m):
        self.x = np.zeros((n, 1))
        self.u = np.zeros((m, 1))
        self.x_next = np.zeros((n, 1))
        self.K = np.zeros((m, n))

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

        self.Q = np.array([[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]])
        self.R = np.array([[1, 0], [0, 1]])

        self.K0 = np.array([[-0.62098405, -0.47989766, -0.12446556],
                            [-0.47989766, -0.47989766, -0.22446556]])
        print("init K is stable: ", self.check_feedback_stabilizable(self.K0))

    def step(self, x, u):
        # assert u.shape[0] == self.B.shape[1]
        # assert x.shape[0] == self.A.shape[0]
        x_next = self.A @ x + self.B @ u
        # check nan
        if np.isnan(x_next).any():
            print("nan")
        return x_next

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


def simulation(lti):
    K = lti.K0
    n = lti.A.shape[0]
    m = lti.B.shape[1]
    print("K is stable: ", lti.check_feedback_stabilizable(K))
    data_vector = np.zeros(0, dtype=data_pair)
    for i in range(1000):
        # random generate u from uniform distribution [-3, 3]
        x = np.random.uniform(-4, 4, (n, 1))
        u = np.random.uniform(-0.1, 0.1, (m, 1))
        x_next = lti.step(x, u)
        data = data_pair(lti.B.shape[0], lti.B.shape[1])
        data.x = x
        data.u = u
        data.x_next = x_next
        data.K = K
        data_vector = np.append(data_vector, data)

    return data_vector

def learning():
    lti = LTI()
    K = lti.K0
    n = lti.A.shape[0]
    m = lti.B.shape[1]
    iteration = 10
    k_vector = np.zeros((n * m, iteration))
    k_vector[:, 0] = K.reshape((n * m, 1))[:, 0]
    for i in range(iteration-1):
        # random generate x0 from uniform distribution [-3, 3]
        data_vector = simulation(lti)
        # solve S from data
        S = solve_S_from_data_collect(data_vector, lti.Q, lti.R)
        # solve K from S
        S_22 = S[n:, n:]
        S_12 = S[:n, n:]
        K = -np.linalg.inv(S_22) @ S_12.T
        k_vector[:, i+1] = K.reshape((n * m, 1))[:, 0]
        lti.update_K0(K)
        print("K = ", K)

    # plot K
    optimal_K = lti.get_optimal_K().reshape((n * m, 1))
    for i in range(k_vector.shape[0]):
        plt.figure()
        plt.plot(k_vector[i, :], label="K_{}".format(i))
        # plot a line
        plt.plot(np.ones((iteration, 1)) * optimal_K[i], label="K_{}_optimal".format(i))
        plt.legend()
        plt.show()




def solve_S_from_data_collect(data_vector, Q, R):
    # S can be solved with least square method
    n = data_vector[0].x.shape[0]
    m = data_vector[0].u.shape[0]
    xi = np.zeros((n + m, 1))   # xi = [x; u]
    zeta = np.zeros((n + m, 1))  # zeta = [x_next; u_next]
    temp = np.kron(xi.T, zeta.T)
    A = np.zeros((data_vector.shape[0], temp.shape[1]))
    b = np.zeros((data_vector.shape[0], 1))
    for i in range(data_vector.shape[0]):
        x = data_vector[i].x
        u = data_vector[i].u
        xi[:n, :] = x
        xi[n:, :] = u
        zeta[:n, :] = data_vector[i].x_next
        u_next = data_vector[i].K @ data_vector[i].x_next
        zeta[n:, :] = u_next
        temp = np.kron(xi.T, xi.T) - np.kron(zeta.T, zeta.T)
        A[i, :] = temp
        b[i, :] = x.T @ Q @ x + u.T @ R @ u
    S = np.linalg.lstsq(A, b, rcond=None)[0]
    S = S.reshape((n + m, n + m))
    return S


if __name__ == '__main__':
    learning()
    lti = LTI()
    print("optimal K = ", lti.get_optimal_K())




