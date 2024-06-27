function noise_M = find_noise_M(W,data, noise_label)
%W: c*d
%data: d*n
%label: c*n

%[nCla,nSam] = size(noise_label);
%noise_M = eye(nCla,nCla);

%% ---------- find the candidate anchor points set------------------
% prediction of W
% [predall1,probablity1,pred1,~] = Predict(W1, data,label);
[predall,probablity,pred,~] = Predict(W, data,noise_label);
[max_probablity,class_index] = max(predall');
% Uniq1 = unique(pred1);
% Uniq = unique(pred);
% % Anchor_index1 = [];
% Anchor_index = [];
% % for i = 1:length(Uniq1)
% %     index = pred1==Uniq1(i);
% %     probablity1_temp = probablity1.*index;
% %     [anchor_prob,anchor_index] = max(probablity1_temp);
% %     Anchor_index1 = [Anchor_index1;anchor_index];
% % end
% for i = 1:length(Uniq)
%     index = pred==Uniq(i);
%     probablity_temp = probablity.*index;
%     [~,anchor_index] = max(probablity_temp);
%     Anchor_index = [Anchor_index;anchor_index];
% end
noise_M = predall(:,class_index)';
% noise_M(pred1(Anchor_index1),:) = predall1(:,Anchor_index1)';
% noise_M(pred2(Anchor_index2),:) = noise_M(pred2(Anchor_index2),:) + predall2(:,Anchor_index2)';
% for i = 1:size(noise_M,1)
%     if sum(noise_M(i,:))>1
%         noise_M(i,:) = noise_M(i,:)/2;
%     end
% end

