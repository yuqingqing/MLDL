%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     code for "Multi-view low-rank dictionary learing"      %
%                   by yqq  2016.1.21                        %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear;
%tr_dat,trls,tt_dat,ttls

% load('E:\yqq\DATABASE\bjut_group_video_clips_imgs\bjut_crowd_video_3view_PCA.mat');
load('E:\yqq\DATABASE\bjut_group_video_clips_imgs\bjut_crowd_video_3view.mat')

bjut_3v{1} = X_V1;
bjut_3v{2} = X_V2(1:419,:);
bjut_3v{3} = X_V3(1:419,:);
bjut_label = label;

for times = 1:20

    idx_rand  = randperm(15);%产生1到15之间的一个随机序列
    idx_test  = idx_rand(1);
    idx_train = idx_rand(2:15);
    
    trls       = [];
    ttls       = [];
    for i = 1:3
        data_temp = bjut_3v{i};
        train_data = [];
        test_data  = [];

        for j = 1:5
            data_t = data_temp(:,find(bjut_label ==  j));   %第j类数据
            test_data  = [test_data,data_t(:,idx_test)];    %随机选 1个测试
            train_data = [train_data,data_t(:,idx_train)];  %剩下的14个训练
            
            if i == 1
                trls = [trls,repmat(j,[1,14])];
                ttls = [ttls,j];
            end
            
        end
        tr_dat{i} = train_data;
        tt_dat{i} = test_data;
    end
    
    r_ratio(times) = MLDL_MASTER(tr_dat,tt_dat,trls,ttls);   %MLDL    
end
save(['E:\yqq\MATLAB\Z_ExperimentResult\bjut_crowd_video_3view_ACC_',num2str(times),'.mat'],'r_ratio');