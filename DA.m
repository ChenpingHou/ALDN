function [W_best,alpha_best,object_value] = DA(vanish_rate,W1,T,data_s2, label_s2,noise_label_s2,alpha_set)
% label:c*n
% feature:d*n
% w:c*d
% alpha_best = 10^[-3:1:3];
% beta_best = 10^[-3:1:3];
%% cross validation
if length(alpha_set) >1
    alpha_num = length(alpha_set);
    n = size(data_s2,2);
    k_fold = 3;
    Indices = crossvalind('Kfold', n, k_fold);
    acc_statistic = zeros(alpha_num);
    for k = 1:k_fold
        x_train = data_s2(:,~(Indices==k));y_train = noise_label_s2(:,~(Indices==k));
        x_test = data_s2(:,Indices==k);y_test = label_s2(:,Indices==k);
        for i = 1:alpha_num
            alpha = alpha_set(i);
            [w_ij,~] = find_best_W(vanish_rate,W1,T,x_train,y_train,alpha);
            [~,~,~,acc_i] = Predict(w_ij,x_test, y_test);
            acc_statistic(i) = acc_statistic(i)+acc_i;
        end
    end
    [index1] = find(acc_statistic==max(acc_statistic));
    alpha_position = index1(1);
    alpha_best = alpha_set(alpha_position);
    [W_best,object_value] = find_best_W(vanish_rate,W1,T, data_s2, noise_label_s2, alpha_best);
else
    alpha_best = alpha_set;
    [W_best,object_value] = find_best_W(vanish_rate,W1,T, data_s2, noise_label_s2, alpha_best);
end
end

function [W_optimal,object_value] = find_best_W(vanish_rate,W1,T,data, noise_label, alpha)
% label:c*n
% feature:d*n
% w:c*d
[nCla,nFea] = size(W1);
nFea_spec = floor(vanish_rate*nFea);
nFea_share = nFea - nFea_spec;
W1_spec = W1(:,1:nFea_spec);
W1_share = W1(:,nFea_spec+1:end);

[nFea2,nSam2] = size(data);
d_new = nFea2 - nFea_share;
W0_d_new = d_new.*(W1_spec*T);
W0 = [W1_share W0_d_new];
cost_old = 0;
cost = 1;
object_value = [];% record the cost at each update iteration
count = 0; % count the running number
[nCla,nFea1] = size(W0);
[nFea2,nSam2] = size(data);
W2 = 0.005*ones(nCla,nFea2); %  Initialise classifier W
loop_max=3000;
eta = exp(-1);
while (abs(cost_old - cost) > 10^-6 && count < loop_max )
    %( abs(cost_old - cost) > 0.0001*cost ) &&
    cost_old = cost;
    count=count+1;
    M = bsxfun(@minus,W2*data,max(W2*data, [], 1));
    M = exp(M);
    p = bsxfun(@rdivide, M, sum(M));
    cost = -1/nSam2 * noise_label(:)' * log(p(:)) +alpha * sum((W2(:)-W0(:)) .^ 2);
    W2_grad = -1/nSam2 * (noise_label - p) * data'  + 2*alpha*(W2-W0);
    
    eta = find_eta(data,noise_label, alpha,W0,W2,W2_grad,cost,eta);
    W2 = W2 - eta*W2_grad;
    object_value = [object_value;cost];
end
W_optimal = W2;
end

function eta_find = find_eta(data,noise_label, alpha,W0,W2,W2_grad,cost_before,eta)
W2_temp = W2 - eta*W2_grad;
[nFea2,nSam2] = size(data);
[nCla,~] = size(W2);
M = bsxfun(@minus,W2_temp*data,max(W2_temp*data, [], 1));
M = exp(M);
p = bsxfun(@rdivide, M, sum(M));
cost_after = -1/nSam2 * noise_label(:)' * log(p(:)) +alpha * sum((W2_temp(:)-W0(:)) .^ 2);
if isnan(cost_after)
    cost_after = 10^8;
end
while cost_after > cost_before
    eta = 0.5*eta;
    W2_temp = W2 - eta*W2_grad;
    M = bsxfun(@minus,W2_temp*data,max(W2_temp*data, [], 1));
    M = exp(M);
    p = bsxfun(@rdivide, M, sum(M));
    cost_after = -1/nSam2 * noise_label(:)' * log(p(:)) +alpha * sum((W2_temp(:)-W0(:)) .^ 2); 
    if isnan(cost_after)
        cost_after = 10^8;
    end
end
eta_find = eta;
end


