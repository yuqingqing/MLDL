function [Di,Ei,Xii] = Update_Di(D,D_label,A,A_label,X,X_label,k,i,par)

%----------------
%   Initialize
%----------------
G         = zeros(size(D{k}(:,D_label{k} == i)));
Xi        = X{k}(:,X_label{k} == i);%Xi 竖着排的[Xi1;Xi2;...;Xic]
ni        = size(Xi,2); %第i类样本数
r1        = ni*i-(ni-1);
r2        = ni*i;
Xii       = Xi(r1:r2,:);

Ei        = zeros(size(A{k}(:,A_label{k} == i)));
T1        = zeros(size(A{k}(:,A_label{k} == i)));
T2        = zeros(size(D{k}(:,D_label{k} == i)));
T3        = zeros(size(Xii));

Ai        = A{k}(:,A_label{k} == i);
Di        = D{k}(:,D_label{k} == i); %300×7
I         = eye(size(Di,2));         %单位矩阵是eye,不是ones I:7×7

delta     = 1e-6;
delta_max = 1e30;
epsilon   = 1e-8;
v         = 1.1;

% 4.
H0 = zeros(size(D{k},1));%方阵
for m = 1:par.M
    if m ~= k
        H0 = H0 + (D{m}*D{m}');
    end
end

converged = 0;
iter      = 1;

while ~converged && iter <= 300
    
    fprintf(['Updating Dictionary,view:' num2str(k) ' class: ',num2str(i),' iter: ',num2str(iter),'\n']);
    %----------------
    %   1.Update Z
    %----------------
    %argmin 1/2||X-U||_F^2 + rho||X||_1
    U1    = Xii + T3/delta;
    rho   = 1/delta;
    Z_new = sign(U1).*max(abs(U1)-rho,0);%|U|:求U的绝对值
    
    %----------------
    %   2.Update Xii
    %----------------
    
    %     Xii_new = (Di'*Di+I).^(-1)*(Di'*(Ai-Ei)+Z_new+((Di'*T1-T3)/delta));
    %------------------------------------------------------------------------
    %     Temp1 = inv(Di'*Di+I);%求矩阵的逆
    %     Temp2 = Di'*(Ai-Ei)+Z_new+((Di'*T1-T3)/delta);
    %     Xii_new = Temp1 * Temp2;
    %------------------------------------------------------------------------
    
    Temp1 = Di' * Di + I;%求矩阵的逆
    Temp2 = Di' * (Ai-Ei) + Z_new + ((Di'*T1-T3)/delta);
    Xii_new = Temp1 \ Temp2;%A的逆矩阵是 A1，则 B/A = B*A1，A\B = A1*B 即：inv(Temp1)*Temp2 <=>Temp1\Temp2;
    
    change(iter,1) = norm((Xii_new-Xii),'inf');
    
    %----------------
    %   3.Update G
    %----------------
    %argmin 1/2||W-C||_F^2 + lambda||W||_*
    lambda1  = par.alpha/delta;
    C        = Di + T2/delta;
    [U2,S,V] = svd(C);
    Sigma    = max(S-lambda1,0);
    G_new    = U2 * Sigma * V';
    
    change(iter,2) = norm((G_new-G),'inf');
    
    %----------------
    %   4.Update Di
    %----------------
    % HY + YQ = V matlab code
    
    H = ( (2 * par.mu * par.beta) / delta ) * H0;
    
    V0 = zeros(size(D{k}(:,D_label{k} == i)));
    for j = 1:par.nClass
        if j ~= i
            nj  = size(X{k}(:,X_label{k} == j),2);%此数据库是
            r1  = nj*j-(nj-1);%nj:第j类样本的个数
            r2  = nj*j;
            Xij = Xi(r1:r2,:);
            Dj  = D{k}(:,D_label{k} == j);
            V0  = V0 + Dj * (Xij * Xii_new');
        end
    end
    V = (2 * par.mu / delta) * (Ai * Xii_new' - V0) + (Ai * Xii_new') - (Ei * Xii_new')...
        + G_new + ((T1 * Xii_new' - T2) / delta);

    Q = (2 * par.mu * (Xii_new * Xii_new')) / delta + (Xii_new * Xii_new') + I;
        
%     Y  = sylvester(H,Q,V);%Solve Sylvester equation HY + YQ = V
%     Y = funcSylvester(H,Q,V);

    V = -V;
    Y = lyap(H,Q,V);
    Di_new = Y;
    
    change(iter,3) = norm((Di_new-Di),'inf');
    
    %----------------
    %   5.Update Ei
    %----------------
    %min lambda||W||_2,1 + 1/2 ||W-U||_F^2
    U3 = Ai - (Di_new * Xii_new) + (T1 / delta);
    lambda2 = par.ita / delta;
    
    Ei_old = Ei;
    for j = 1:size(U3,2)
        uj = U3(:,j);
        Temp3 = norm(uj);
        if(lambda2 < Temp3)
            Ei(:,j) = (Temp3 - lambda2) / Temp3 * uj;
        else
            Ei(:,j) = zeros(size(uj));
        end
    end
       
    change(iter,4) = norm((Ei-Ei_old),'inf');
    Ei_new = Ei;
    
    %--------------------------
    %   6.Update T , delta
    %--------------------------
    T1 = T1 + delta * (Ai - Di_new * Xii_new - Ei_new);
    T2 = T2 + delta * (Di_new - G_new);
    T3 = T3 + delta * (Xii_new - Z_new);
    
    delta = min(v * delta, delta_max);
    
    Z   = Z_new;
    Xii = Xii_new;
    G   = G_new;
    Di  = Di_new;
    Ei  = Ei_new;
    
    f1 = norm((Di - G),'inf');
    f2 = norm((Ai - Di * Xii - Ei),'inf');
    f3 = norm((Xii - Z),'inf');
    
    if (f1 < epsilon) && (f2 < epsilon) && (f3 < epsilon)
        converged = 1;
    end;
    
    iter = iter + 1;
end

figure(1);
subplot(2,2,1);plot(change(:,1),'-');title('changing-xii');
subplot(2,2,2);plot(change(:,2),'-');title('changing-Di');
subplot(2,2,3);plot(change(:,3),'-');title('changing-G');
subplot(2,2,4);plot(change(:,4),'-');title('changing-Ei');
end