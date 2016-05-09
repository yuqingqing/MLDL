%---------------------------------------------------
%                   MLDL MASTER
%---------------------------------------------------
function [r_ratio] = MLDL_MASTER(tr_dat,tt_dat,trls,ttls)
iter         =  1;
converged    =  0;

par.nClass    =   length(unique(trls));
par.lambda1   =   0.005;
par.lambda2   =   0;
par.nIter     =   300;       %迭代次数
par.M         =   length(tr_dat); %M个视角

par.alpha     =   1;       %参数设置
par.ita       =   0.01;
par.mu        =   1;
par.beta      =   0.05;
par.epsilon   =   1e-8;

par.gama = 0.05;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                    sample
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = tr_dat;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              initialize dict
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k = 1:par.M
    Dict_ini   = [];
    Dlabel_ini = [];
    A_label{k} = trls;
    for ci = 1:par.nClass
        fprintf(['Initializing Dict: View ' num2str(k) '  Class ' num2str(ci) '\n']);
        cdat          =    A{k}(:,A_label{k} == ci);
        dict          =    Initialize_Di(cdat);
        Dict_ini      =    [Dict_ini dict];
        Dlabel_ini    =    [Dlabel_ini repmat(ci,[1 size(dict,2)])];
    end
    Dict{k}   = Dict_ini;
    Dlabel{k} = Dlabel_ini;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              initialize coef
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ini_par.tau        =     par.lambda1;
ini_par.lambda     =     par.lambda2;

for k = 1:par.M
    ini_ipts.D = Dict{k};
    
    X{k}       = zeros(size(Dict{k},2),size(A{k},2));      %字典样本数×训练样本数 700×700
    X_label{k} = trls;
    
    if size(Dict{k},1) > size(Dict{k},2)
        ini_par.c        =    1.05*eigs(Dict{k}'*Dict{k},1);%特征值or特征向量
    else
        ini_par.c        =    1.05*eigs(Dict{k}*Dict{k}',1);
    end
    
    for ci =  1:par.nClass
        fprintf(['Initializing Coef: View ' num2str(k) '  Class ' num2str(ci) '\n']);
        ini_ipts.X      =    A{k}(:,A_label{k}==ci);
        [ini_opts]      =    FDDL_INIC (ini_ipts,ini_par);
        X{k}(:,X_label{k} == ci) = ini_opts.A;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           Main loop of Dictionary Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while iter <= par.nIter && ~converged
    
    for k = 1:par.M
        
        %--------------------------------------------------
        %  updating the coefficient IPM algorithm,FDDL[4]
        %--------------------------------------------------
        [X_new{k}] = Update_X(A{k},A_label{k},Dict{k},Dlabel{k},X{k},k,par);
        
        %--------------------------------------------------
        %      updating the dictionary ALM algorithm
        %--------------------------------------------------
        [D_new{k},Error{k},X_new{k}] = Update_D(Dict,Dlabel,A,A_label,X_new,X_label,k,par);
        
    end
    
    e1 = 0;
    e2 = 0;
    
    for k = 1:par.M %M个视角的误差
        X_E{k} = X_new{k} - X{k};
        D_E{k} = D_new{k} - Dict{k};
        e1 = e1 + norm(X_E{k},'inf');
        e2 = e2 + norm(D_E{k},'inf');
    end
    
    x_e(iter) = e1;
    d_e(iter) = e2;
    
    if e1 < par.epsilon && e2 < par.epsilon
        converged = 1;
    end
    
    X    = X_new;
    Dict = D_new;
    
    iter = iter + 1;
end

figure(1);
subplot(1,2,1);plot(x_e(:,1),'-');title('changing-X');
subplot(1,2,2);plot(d_e(:,1),'-');title('changing-D');

[r_ratio,w_ratio] = MLDL_Classification(tt_dat,ttls,Dict,Dlabel{1},par);

fprintf(['正确率：',num2str(r_ratio),'\n']);
% disp('错误率：',w_ratio);
return;