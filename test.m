clear;



load('AR_EigenFace');%tr_dat,trls,tt_dat,ttls
load('.\temp_value\var_01.mat');

% par.alpha     =   1;       %≤Œ ˝…Ë÷√
% par.ita       =   0.01;
% par.mu        =   1;
% par.beta      =   0.05;
% par.M         =   3;
% par.nClass    =   size(unique(ttls),2)
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %                    sample
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for k = 1:par.M
%     A{k}       = tr_dat;
%     A_label{k} = trls;
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %              initialize dict
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for k = 1:par.M    
%     Dict_ini   = [];
%     Dlabel_ini = [];
%     for ci = 1:par.nClass
%         fprintf(['Initializing Dict: View ' num2str(k) '  Class ' num2str(ci) '\n']);
%         cdat          =    A{k}(:,A_label{k}==ci);
%         dict          =    Initialize_Di(cdat);
%         Dict_ini      =    [Dict_ini dict];
%         Dlabel_ini    =    [Dlabel_ini repmat(ci,[1 size(dict,2)])];
%     end
%     Dict{k}   = Dict_ini;
%     Dlabel{k} = Dlabel_ini;
% end
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %             simulate coep
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for k = 1:par.M
%     X{k}       = rand(size(tr_dat,2));
%     X_label{k} = trls;
% end
% 
k = 1;

Update_D(Dict,Dlabel,A,A_label,X,X_label,k,par);