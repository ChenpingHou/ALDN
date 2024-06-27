function T = Optimal_Transport(vanish_rate,data_s1,data_s2)
% This code is to learn the optial transport matrix T.
%% Input
% feature:n*d
%%
[nSam1,nFea1]=size(data_s1); 
nFea_spec1 = floor(vanish_rate*nFea1);
nFea_share = nFea1 - nFea_spec1;
data_s1_spec = data_s1(:,1:nFea_spec1);
data_s1_share = data_s1(:,nFea_spec1+1:end);
data_s2_share = data_s2(:,1:nFea_share);  %n2*d1
data_s2_spec = data_s2(:,nFea_share+1:end);
M1 = OMP(data_s1_share,data_s1_spec,40);   %use data_s12 to represent data_s11
M2 = OMP(data_s2_share,data_s2_spec,40); %use data_s21 to represent data_s22
% compute the feature transition cost matrix cost_Q using M1 and M2
cost_Q = sqrt(-bsxfun(@minus,bsxfun(@minus,2*M1'*M2,sum(M1.^2,1)'),sum(M2.^2,1)));
% compute the optimal transport matrix T
% 
[N1, N2] = size(cost_Q);
mu_1 = ones(N1,1)/N1;
mu_2 = ones(N2,1)/N2; 
%set the normalized marginal probability mass vectors \mu_1 and \mu_2
% according to Ye Hanjia(TPAMI 2021)
rho = 1./N1;
max_iter = 20000;
[V, T] = OptimalTransport_IBP_Sinkhorn(cost_Q, mu_1, mu_2, rho, max_iter);




