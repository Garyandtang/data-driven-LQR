import control as ct
import numpy as np
import matplotlib.pyplot as plt

class data:
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
        print("K = ", -K)
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



if __name__ == '__main__':
    lti = LTI()
    # lti.check_controllable()
    # K = lti.get_optimal_K()
    # print(lti.check_feedback_stabilizable(K))
