
clc;clear;

load('coil_20_3v_norm.mat');%tr_dat,trls,tt_dat,ttls

times = 1;
for t = 1:times
%--------------------
%     trad,testd
%--------------------
idx_rand = randperm(72);%randperm(n)产生1到n的整数的无重复的随机排列

for k = 1:3
    trad{k}  = [];
    tradl = [];
    testd{k} = [];
    testl = [];
    for c = 1:20 %20 objects
        object     = coil_20_3v{k}(:,coil_20_label == c);
        train_temp = object(:,idx_rand(1:36));
        trad{k}    = [trad{k},train_temp];
        tradl   = [tradl,repmat(c,1,36)];
        
        test_temp = object(:,idx_rand(37:72));
        testd{k}  = [testd{k},test_temp];
        testl  = [testl,repmat(c,1,36)];
    end
end

MLDL_MAIN;

end