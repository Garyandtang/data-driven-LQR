x = sym("x",[3,1]);
Q = sym("Q",[3,3])
R = sym("R",[2,2])
K = sym("K",[2,3])
d = sym("d",[2,1])
A = sym("A",[3,3])
B = sym("B",[3,2])
x = sym("x");
Q = sym("Q")
Qf = sym("Qf")
R = sym("R")
K = sym("K")
d = sym("d")
A = sym("A")
B = sym("B")
u = sym("B")

cost = x*Q * x + u * R * u + (A *x + B * u) * Qf * (A *x + B * u)

jacobian(cost, u)
