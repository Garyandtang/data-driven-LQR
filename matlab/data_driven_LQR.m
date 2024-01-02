v = 1;
w = 1;
dt = 0.02;

l = 0.0518;
r = 0.1908;
adj = [0, w, 0;
       -w, 0, v;
       0, 0, 0];
A = eye(3) + dt * adj;
B = dt * [r/2, r/2;
           0, 0;
           -r/l, r/l];


c =-dt* [v, 0, w]';

Q = 200* eye(3)
R = 0.2 * eye(2);

A_ = zeros(6,6);
B_ = zeros(6, 2);
A_(1:3, 1:3) = A;
A_(1:3, 4:6) = eye(3);
A_(4:6, 4:6) = eye(3);
B_(1:3, 1:2) = B;
Q_ = zeros(6,6);
Q_(1:3, 1:3) = Q;
S_next = Q_;
K_containner = [];

K_optimal = -dlqr(A,B,Q,R)
K = -dlqr(A,B,Q,R)
k = -rand(2,1)
k = -pinv(B) * c
k_ini = k
for i = 1:20000
    K_ = -inv(R + B_' * S_next * B_) * B_' * S_next * A_;
    S = A_' * S_next * A_ + A_' * S_next * B_ * K_;
    S_next = S;
    K_containner = [K_; K_containner];
end

x = [2;3;14;c];
cost = 0
for i = 1 : 20000
    K = K_containner(2*i-1 : 2*i,:);
    u = K * x;
    cost = x' * Q_ * x + u' * R * u + cost;
    x = (A_ + B_*K) * x
end
cost
x = [2;3;14];
cost_ = 0;
for i = 1 : 200000
    u = K_optimal * x + k;
    cost_ = x' * Q * x + u' * R * u + cost_;
    x = A * x + B * u + c;
end
cost_
% 
% K  = rand(2,3)



n = size(B, 1);
m = size(B, 2);



% x =  -10 + 20 * rand(3, 1);
x_vec = [];
x_next_vec = [];
u_vec = [];
for i = 1 : 500
    x =  -10 + 20 * rand(3, 1);
    u =  -10 + 20*rand(2, 1)
    x_next = A * x + B * u + c;
    x_vec = [x_vec, x];
    x_next_vec = [x_next_vec, x_next];
    u_vec = [u_vec, u];

end

iteration = 1;

for i = 1 : iteration
    
end

for i = 1 : 20
    i
    xi = zeros(2*n+m, 1);
    temp = kron(xi', xi');
    A_s = zeros(size(x_vec, 2), size(temp,2));
    b = zeros(size(x_vec, 2),1);
    for i = 1 : size(x_vec, 2)
        x = x_vec(:,i);
        x_next = x_next_vec(:, i);
        u = u_vec(:,i)- k;
        u_next = K * x_next;
        xi = [x;u;c];
        zeta = [x_next; u_next;c];
        temp = kron(xi', xi') - kron(zeta', zeta');
        A_s(i,:) = temp;
        b(i, :) = x' * Q * x + u' * R * u;
    end
    S = pinv(A_s) *b;

    S = reshape(S, 2*n+m, 2*n+m);
    
    S_11 = S(1:n, 1:n);
    S_12 = S(1:n, n+1 : n+m);
    S_22 = S(n+1:n+m, n+1:n+m);
    S_23 = S(n+1:n+m , n+m+1 : 2*n + m);
    
    K_cal = -inv(S_22) * S_12'
    K = K_cal;
    K_optimal
    B_cal = A * inv(S_11 - Q) * S_12
    B
    k = -pinv(B_cal) * c
    k = - inv(S_22) * S_23 * c
    aaa = 1;
end

x = [1;1;1];
x_container = x;
cost = 0;
for i = 1 : 1000
    u = K_optimal * x + k_ini;
    x = A * x + B * u + c;
    x_container = [x_container, x];
    cost = x'*Q*x + u' * R * u + cost;
end
plot(x_container(1, :))
cost

