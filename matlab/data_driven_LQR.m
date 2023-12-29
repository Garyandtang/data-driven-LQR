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

Q = 2* eye(3)
R = 2 * eye(2);

K_optimal = -dlqr(A,B,Q,R)
K = -dlqr(A,B,Q,R)
% 
% K  = rand(2,3)
k = -rand(2,1)
k = -pinv(B) * c
k_ini = k


n = size(B, 1);
m = size(B, 2);



% x =  -10 + 20 * rand(3, 1);
x_vec = [];
x_next_vec = [];
u_vec = [];
for i = 1 : 50
    x =  -100 + 200 * rand(3, 1);
    u =  -100 + 200*rand(2, 1)
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
%     k = - inv(S_22) * S_23 * c
    aaa = 1;
end

x = rand(3,1);
x_container = x;
for i = 1 : 100
    u = K_cal * x + k;
    x = A * x + B * u + c;
    x_container = [x_container, x];
end
plot(x_container(1, :))

