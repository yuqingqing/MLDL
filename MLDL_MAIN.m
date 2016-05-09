%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     code for "Multi-view low-rank dictionary learing"      %
%                   by yqq  2016.1.21                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear;

% load('AR_EigenFace'); %tr_dat,trls,tt_dat,ttls

load('E:\yqq\MATLAB\MLDL\CHUK_2VIEW_472.mat');
% CHUK


label_unique = unique(label);
perc = 1;
nGroup = size(View{1},2);
% generate data index for training
for n = 1 : length(label_unique)
    class_num(n) = length(find(label == label_unique(n)));
end
% random select one out to test //8 to train
rndIdx = []; 
for n = 1 : length(label_unique)
    K = round(class_num(n) * perc);
    idx_rand = randperm(class_num(n));
    idx_curGr = find(label == label_unique(n));
    rndIdx = [rndIdx,idx_curGr(idx_rand(2:K))];
end
restIdx = setdiff(1:nGroup, rndIdx);
rndIdx = [];
for j=1:8
    idx_class = setdiff(find(label == j),restIdx(j));
    rndIdx = [rndIdx,idx_class(1:8)];
end

tradl = label(rndIdx);
testl = label(restIdx);

for i=1:2
    trad{i}  = View{i}(:, rndIdx);
    testd{i} = View{i}(:, restIdx);
end


% %tr_dat preprocess
% load('E:\yqq\MATLAB\QMUL_FACES\QMUL_31_noise40_0.mat');
% % QMUL
% for i=1:3
%     trad{i} = [];
%     tradl =[];
%     for j=1:31 
%     testd{i}(:,j) = train_data_noise40{i}(:,j*7);
%     testl(1,j) = j;
%     
%     trad{i} = [trad{i},train_data_noise40{i}(:,j*7-6:j*7-1)];
%     tradl = [tradl,train_label_noise40(:,j*7-6:j*7-1)];     
%     end
% end

% %COIL-20
% trad = tr_dat;
% tradl = trls;
% testd = tt_dat;
% testl = ttls; 

iter         =  0;
converged    =  0;

par.nClass    =   length(unique(tradl));
par.lambda1   =   0.005;
par.lambda2   =   0;
par.nIter     =   2;      %迭代次数
par.M         =   length(trad); %M个视角

par.alpha     =   1;       %参数设置
par.ita       =   0.01;
par.mu        =   1;
par.beta      =   0.05;
par.epsilon   =   1e-8;

par.gama = 0.05;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                    sample
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A = trad;    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              initialize dict
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k = 1:par.M    
    Dict_ini   = [];
    Dlabel_ini = [];
    A_label{k} = tradl;
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
    X_label{k} = tradl;
    
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
        
        %-------------------------------------------------
        %      updating the dictionary ALM algorithm
        %-------------------------------------------------
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
    
    if e1 < par.epsilon && e2 < par.epsilon
        converged = 1;
    end
    
    X    = X_new;
    Dict = D_new;
    
    iter = iter + 1;
end

[r_ratio,w_ratio] = MLDL_Classification(testd,testl,Dict,Dlabel{1},par);

fprintf(['正确率：',num2str(r_ratio),'\n']);
% disp('错误率：',w_ratio);
