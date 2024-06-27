function [W_best,alpha_best,object_value] = noise_softmax(data, label, noise_label, alpha_set,noise_M)
% label:c*n
% feature:d*n
% w:c*d
% alpha_best = 0.0001;
% beta_best = 0.01;
%% cross validation
noise_M = inv(noise_M);
if length(alpha_set) >1
    alpha_num = length(alpha_set);
    n = size(data,2);
    k_fold = 3;
    Indices = crossvalind('Kfold', n, k_fold);
    acc_statistic = zeros(alpha_num,1);
    for k = 1:k_fold
        x_train = data(:,~(Indices==k));y_train = noise_label(:,~(Indices==k));
        x_test = data(:,Indices==k);y_test = label(:,Indices==k);
        for i = 1:alpha_num
            alpha = alpha_set(i);
            [w_i,~] = find_best_w(x_train,y_train,alpha,noise_M);
            [~,~,~,acc_i] = Predict(w_i,x_test, y_test);
            acc_statistic(i) = acc_statistic(i)+acc_i;
        end
    end
    index = find(acc_statistic==max(acc_statistic));
    alpha_position = index(1);
    alpha_best = alpha_set(alpha_position);
    [W_best,object_value] = find_best_w(data, noise_label, alpha_best,noise_M);
else
    alpha_best = alpha_set;
    [W_best,object_value] = find_best_w(data, noise_label, alpha_best,noise_M);
end

end

function [w_best,object_value] = find_best_w(data, noise_label, alpha,noise_M)
%set old cost and new cost value
cost_old = 0;
cost = 1;
object_value = [];% record the cost at each update iteration
count = 0; % count the running number
[nCla,~] = size(noise_label);
[nFea,nSam] = size(data);
W = 0.005*ones(nCla,nFea); %  Initialise classifier W
loop_max=2000;
eta = exp(-1);
while (abs(cost_old - cost) > 10^-5 && count < loop_max )
    %( abs(cost_old - cost) > 0.0001*cost ) &&
    cost_old = cost;
    M = bsxfun(@minus,W*data,max(W*data, [], 1));
    M = exp(M);
    p = bsxfun(@rdivide, M, sum(M));
    Ide = eye(nCla,nCla);
    W_grad = zeros(nCla,nFea);
    cost = 0;
    for c = 1:nCla
        Ec = Ide(c,:);
        EC = repmat(Ec',1,nSam); %c*n
        pre_label = p; %c*n
        y_tilde = vec2ind(noise_label);
        noise_M_c = noise_M(:,c);
        m = noise_M_c(y_tilde);
        pre_label = pre_label.*m';
        EC = EC.*m';
        W_grad = W_grad + (EC - pre_label)*data';
        cost = cost + EC(:)' * log(p(:));%%%%%% need to be verified, important!
    end
    cost = -1/nSam * cost + alpha * sum(W(:) .^ 2);
    W_grad = -1/nSam * W_grad + 2*alpha*W;
    eta = find_eta_noise(data,noise_label, alpha,W,W_grad,cost,eta,nSam,nCla,noise_M);
    W = W - eta*W_grad;
    object_value = [object_value;cost];
    count=count+1;
end
w_best = W;
end

function eta_find = find_eta_noise(data,noise_label, alpha,W,W_grad,cost_before,eta,nSam,nCla,noise_M)
W1 = W - eta*W_grad;
M = bsxfun(@minus,W1*data,max(W1*data, [], 1));
M = exp(M);
p = bsxfun(@rdivide, M, sum(M));
Ide = eye(nCla,nCla);
%W_grad = zeros(nCla,nFea);
cost = 0;
for c = 1:nCla
    Ec = Ide(c,:);
    EC = repmat(Ec',1,nSam); %c*n
    %pre_label = p; %c*n
    y_tilde = vec2ind(noise_label);
    noise_M_c = noise_M(:,c);
    m = noise_M_c(y_tilde);
    % pre_label = pre_label.*m';
    EC = EC.*m';
    % W_grad = W_grad + (EC - pre_label)*data';
    cost = cost + EC(:)' * log(p(:));%%%%%% need to be verified, important!
end
cost_after = -1/nSam * cost + alpha * sum(W1(:) .^ 2);
if isnan(cost_after)
    cost_after = 10^8;
end
while cost_after > cost_before
    eta = 0.5*eta;
    W1 = W - eta*W_grad;
    M = bsxfun(@minus,W1*data,max(W1*data, [], 1));
    M = exp(M);
    p = bsxfun(@rdivide, M, sum(M));
    Ide = eye(nCla,nCla);
    %W_grad = zeros(nCla,nFea);
    cost = 0;
    for c = 1:nCla
        Ec = Ide(c,:);
        EC = repmat(Ec',1,nSam); %c*n
        %pre_label = p; %c*n
        y_tilde = vec2ind(noise_label);
        noise_M_c = noise_M(:,c);
        m = noise_M_c(y_tilde);
        % pre_label = pre_label.*m';
        EC = EC.*m';
        % W_grad = W_grad + (EC - pre_label)*data';
        cost = cost + EC(:)' * log(p(:));%%%%%% need to be verified, important!
    end
    cost_after = -1/nSam * cost + alpha * sum(W1(:) .^ 2);
    if isnan(cost_after)
        cost_after = 10^8;
    end
end
eta_find = eta;
end