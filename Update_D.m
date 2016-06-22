function [D_k,E_k,X_k] = Update_D(D,D_label,A,A_label,X,X_label,k,par)

    D_k = [];
    E_k = [];
    X_k = [];
    for ci = 1:par.nClass
        
        X_k_i  =  X{k}(:,X_label{k} == ci);%Xi 竖着排的[Xi1;Xi2;...;Xic]
        ni     =  size(X_k_i,2); %第i类样本数
        r1     =  ni*ci-(ni-1);
        r2     =  ni*ci;
        
        fprintf(['Updating Dictionary,view ' num2str(k) ' class: ' num2str(ci) '\n']);
        [D_k_i,E_k_i,X_k_ii] = Update_Di(D,D_label,A,A_label,X,X_label,k,ci,par);
        
        D_k = [D_k D_k_i];
        E_k = [E_k E_k_i];
        
        X_k_i(r1:r2,:) = X_k_ii;
        X_k = [X_k X_k_i];
    end    

return;