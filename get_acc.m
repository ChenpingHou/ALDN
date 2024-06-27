function acc=get_acc(w,testdata,testlabel)
%此函数为求出线性分类器w在测试集上的分类精度acc
%输入：
%    w:cxd,线性分类器
%    testdata：nxd 测试数据特征矩阵
%    testlabel：nx1 测试数据标签
%输出：
%    acc：在测试集上的分类准确率
dim=size(w,2);
y=w*testdata(:,1:dim)';%若挑选标注的数据较少，则可能在数据维度增大后模型没有进行更新，导致最终的模型w_t的维度与测试数据维度不一致。
[probablity ,pred] = max(y);
acc = mean(testlabel(:) == pred(:));