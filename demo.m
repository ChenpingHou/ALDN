%% demo of the ALDN
% This is the code for the paper "Adaptive Learning for Dynamic Features and
% Noisy Labels". 
% If there are any problems, please feel free to contact
% Chenping Hou (hcpnudt@hotmail.com).
%% generate the dataset

num_cir = 30;
vanish_rate = 9/11;
% ratio_select = 0.5;
%noise_rate = 0.4;
alpha_set=10.^[-3:3]; % parameter
beta_set=10.^[-3:3];  % parameter
[nSam,nFea] = size(data);
len_Fea = floor(0.55*nFea);
% stdX = std(data); % standard deviation
% idx1 = stdX~=0;
% centrX = data-repmat(mean(data),size(data,1),1);
% data(:,idx1) = centrX(:,idx1)./repmat(stdX(:,idx1),size(data,1),1);
% data = (data-repmat(mean(data),size(data,1),1))./repmat(std(data),size(data,1),1);
% data = data./repmat(sqrt(sum(data.*data,2)),1, size(data,2));
x_past = data(:,1:len_Fea);
x_new = data(:,len_Fea+1:end);
num_cla = length(unique(label));
label(label ==0)=num_cla;
c = 1;

Acc_ours_reg1 = [];Acc_ours_reg2 = [];

train_data = []; train_label = [];
test_data = [];test_label=[];
train_data_s1 = [];train_label_s1 = [];
train_data_s2_1 = [];train_label_s2_1 = [];
train_data_s2_2 = [];train_label_s2_2 = [];
for i=1:num_cla
    A=data(label == i,:);
    B=label(label == i);
    len = length(B);
    ratio1 = floor(0.5*len);
    ratio2 = floor(0.55*len);
    ratio3 = floor(0.6*len);
    train_data_s1 = [train_data_s1; A(1:ratio1,:)];
    train_label_s1 = [train_label_s1; B(1:ratio1)];
    train_data_s2_1 = [train_data_s2_1; A(ratio1+1:ratio2,:)];
    train_label_s2_1 = [train_label_s2_1; B(ratio1+1:ratio2)];
    train_data_s2_2 = [train_data_s2_2; A(ratio2+1:ratio3,:)];
    train_label_s2_2 = [train_label_s2_2; B(ratio2+1:ratio3)];
    test_data = [test_data; A(ratio3+1:end,:)];
    test_label = [test_label; B(ratio3+1:end)];
end

train_data_s1 = train_data_s1(:,1:size(x_past,2));
len_vanish = floor(vanish_rate*size(x_past,2));
len_survive = size(x_past,2) - len_vanish;
spec_data_s1 = train_data_s1(:,1:len_vanish);
share_data_s1 = train_data_s1(:,len_vanish+1:end);
train_data_s2_1 = train_data_s2_1(:,len_vanish+1:end);
train_data_s2_2 = train_data_s2_2(:,len_vanish+1:end);
test_data = test_data(:,len_vanish+1:end);
train_data_s2_all = [train_data_s2_1;train_data_s2_2];
train_label_s2_all = [train_label_s2_1;train_label_s2_2];
%noise_train_label_s2_2 = NoiseGenerate(train_label_s2_2,1);
noise_train_label_s2_all = [train_label_s2_1;noise_train_label_s2_2];

% change the label vector to 0-1 matrix:
test_label = full(ind2vec(test_label'));
train_label_s1 = full(ind2vec(train_label_s1'));
%% select the best parameter on the whole data_s2
% r = 1;
% INDEX_clean=[];
% INDEX_noise=[];
% while r <= 100
%     [~, ~,INdex_clean] = Clean_slection(train_data_s2_1,train_label_s2_1,1-noise_rate);
%     [~, ~,~,INdex_noise] = Slection(train_data_s2_2,train_label_s2_2,noise_train_label_s2_2,noise_rate);
%     INDEX_clean = [INDEX_clean;INdex_clean];
%     INDEX_noise = [INDEX_noise;INdex_noise];
%     r = r + 1;
% end
% Rand_index = randperm(100,num_cir);
% INDEX_cl = INDEX_clean(Rand_index,:);
% INDEX_noi = INDEX_noise(Rand_index,:);
[W1,~,object_direct1] = mysoftmax(train_data_s1',train_label_s1,train_label_s1,alpha_set);
while c <= num_cir
    % randomly seclect data from data_s2_all
    train_data_s2 = [train_data_s2_1(INDEX_cl(c,:),:);train_data_s2_2(INDEX_noi(c,:),:)];
    train_label_s2 = [train_label_s2_1(INDEX_cl(c,:));train_label_s2_2(INDEX_noi(c,:))];
    noise_train_label_s2 = [train_label_s2_1(INDEX_cl(c,:));noise_train_label_s2_2(INDEX_noi(c,:))];
    share_data_s2 = train_data_s2(:,1:len_survive);
    % change the label vector to 0-1 matrix:
    train_label_s2 = full(ind2vec(train_label_s2'));
    noise_train_label_s2 = full(ind2vec(noise_train_label_s2'));
    T = Optimal_Transport(vanish_rate,train_data_s1,train_data_s2); % the optimal transport matrix
    %% train & test
    fprintf('Iteration: c=%f \n',c)
    [W_DA,~,~] = DA(vanish_rate,W1,T,train_data_s2', train_label_s2,noise_train_label_s2,alpha_set);
    noise_M_DA =  find_noise_M(W_DA,train_data_s2', noise_train_label_s2);
    [W_ours_reg1,~] = Ours_reg1(vanish_rate,noise_M_DA,W1,T,train_data_s2', train_label_s2,noise_train_label_s2,alpha_ours_reg1);
    [~,~,pred_ours_reg1,acc_ours_reg1] = Predict(W_ours_reg1, test_data',test_label);
    Acc_ours_reg1 = [Acc_ours_reg1;acc_ours_reg1];

    [W_IA,~,~] = IA(vanish_rate,W1,T,train_data_s2', train_label_s2,noise_train_label_s2,alpha_set);        
    noise_M_IA =  find_noise_M(W_IA,train_data_s2', noise_train_label_s2);                      
    [W_ours_reg2,~] = Ours_reg2(vanish_rate,noise_M_IA,W1,T,train_data_s2', train_label_s2,noise_train_label_s2,alpha_ours_reg2);
    [~,~,pred_ours_reg2,acc_ours_reg2] = Predict(W_ours_reg2, test_data',test_label);
    Acc_ours_reg2 = [Acc_ours_reg2;acc_ours_reg2];
        
    c = c+1;
end
mAcc_ours_reg1 = mean(Acc_ours_reg1);
mAcc_ours_reg2 = mean(Acc_ours_reg2);