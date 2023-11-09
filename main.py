import control as ct
import numpy as np
import matplotlib.pyplot as plt

class data_pair:
    def __init__(self, n, m):
        self.x = np.zeros((n, 1))
        self.u = np.zeros((m, 1))
        self.x_next = np.zeros((n, 1))
        self.K = np.zeros((m, n))


class LTI:
    def __init__(self):
        self.A = np.array([[0, 1], [-1, -1]])
        self.B = np.array([[0], [1]])

        self.Q = np.array([[1, 0], [0, 1]])
        self.R = np.array([[1]])

        self.K0 = np.array([[0.7, 0.5]])
        print("init K is stable: ", self.check_feedback_stabilizable(self.K0))

    def step(self, x, u):
        return self.A @ x + self.B @ u

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


def main():
    # LQR
    A = np.array([[0, 1], [-1, -1]])
    B = np.array([[0], [1]])
    Q = np.array([[1, 0], [0, 1]])
    R = np.array([[1]])
    K, S, E = ct.lqr(A, B, Q, R)
    print("K = ", K)

    # Simulation
    x0 = np.array([[1], [0]])

def simulation(x0, lti):
    K = lti.K0
    print("K is stable: ", lti.check_feedback_stabilizable(K))
    x = x0
    data_vector = np.zeros(0, dtype=data_pair)
    for i in range(100):
        u = np.array([[0.5 * np.sin(np.pi*i/40) + 0.3 * np.sin(np.pi*i/5)]])
        x_next = lti.step(x, u)
        data = data_pair(lti.B.shape[0], lti.B.shape[1])
        data.x = x
        data.u = u
        data.x_next = x_next
        data.K = K
        x = x_next
        data_vector = np.append(data_vector, data)

    return data_vector

def learning():
    lti = LTI()
    K = lti.K0
    n = lti.A.shape[0]
    m = lti.B.shape[1]
    iteration = 100
    k_vector = np.zeros((n * m, iteration))
    k_vector[:, 0] = K.reshape((n * m, 1))[:, 0]
    for i in range(iteration-1):
        # random generate x0 from uniform distribution [-3, 3]
        x0 = np.random.uniform(-3, 3, (n, 1))
        print("x0 = ", x0)
        data_vector = simulation(x0, lti)
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
    plt.figure()
    plt.plot(k_vector[0, :], label="K11")
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
    #     # random generate x0 from uniform distribution [-3, 3]
    #     x0 = np.random.uniform(-3, 3, (n, 1))
    #     print("x0 = ", x0)
    #     data_vector = simulation(x0, lti)



