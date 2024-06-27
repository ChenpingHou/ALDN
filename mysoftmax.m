function [W_best,alpha_best,object_value] = mysoftmax( x,y,noise_y,alpha_set)
% 梯度下降法实现softmax多分类
% feature: d*n
% y: c*n
% classifier W: c*d
% alpha_best = 0.0001;
%% cross validation
if length(alpha_set)>1
    alpha_num = length(alpha_set);
    n = size(x,2);
    k_fold = 3;
    Indices = crossvalind('Kfold', n, k_fold);
    acc_statistic = zeros(alpha_num,1);
    for k = 1:k_fold
        x_train = x(:,~(Indices==k));y_train = noise_y(:,~(Indices==k));
        x_test = x(:,Indices==k);y_test = y(:,Indices==k);
        for i = 1:alpha_num
            alpha = alpha_set(i);
            [w_i,~] = find_best_w(x_train,y_train,alpha);
            [~,~,~,acc_i] = Predict(w_i,x_test, y_test);
            acc_statistic(i) = acc_statistic(i)+acc_i;
        end
    end
    index = find(acc_statistic==max(acc_statistic));
    alpha_position = index(1);
    alpha_best = alpha_set(alpha_position);
    [W_best,object_value] = find_best_w( x,noise_y, alpha_best);
else
    alpha_best = alpha_set;
    [W_best,object_value] = find_best_w( x,noise_y, alpha_best);
end
end

function [w_best,object_value] = find_best_w( x,y, alpha)
[n_Fea,n_Sam] = size(x);
nCla = size(y,1);
W = 0.005*ones(nCla,n_Fea);
count=0;        % count the running number
%set old cost and new cost value
cost_old=0;
cost=1;
object_value=[];% record the cost at each update iteration
loop_max=3000;
eta = exp(-1);
while (abs(cost_old - cost) > 10^-6 && count < loop_max )
    %( abs(cost_old - cost) > 0.0001*cost ) &&
    cost_old = cost;
    M = bsxfun(@minus,W*x,max(W*x, [], 1));
    M = exp(M);
    p = bsxfun(@rdivide, M, sum(M));
    cost = -1/n_Sam * y(:)' * log(p(:)) + alpha * sum(W(:) .^ 2);
    W_grad = -1/n_Sam * (y - p) * x' + 2*alpha * W;
    eta = find_eta(x,y, alpha,W,W_grad,cost,eta);
    W = W - eta*W_grad;  
    object_value=[object_value;cost];
    count=count+1;  
end
w_best = W;
end

function eta_find = find_eta(x,y, alpha,W,W_grad,cost_before,eta)
[n_Fea,n_Sam] = size(x);
W1 = W - eta*W_grad;
M = bsxfun(@minus,W1*x,max(W1*x, [], 1));
M = exp(M);
p = bsxfun(@rdivide, M, sum(M));
cost_after = -1/n_Sam * y(:)' * log(p(:)) + alpha * sum(W1(:) .^ 2);
while cost_after > cost_before
    eta = 0.5*eta;
    W1 = W - eta*W_grad;
    M = bsxfun(@minus,W1*x,max(W1*x, [], 1));
    M = exp(M);
    p = bsxfun(@rdivide, M, sum(M));
    cost_after = -1/n_Sam * y(:)' * log(p(:)) + alpha * sum(W1(:) .^ 2);
end
eta_find = eta;
end
