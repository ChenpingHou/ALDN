function acc=get_acc(w,testdata,testlabel)
%�˺���Ϊ������Է�����w�ڲ��Լ��ϵķ��ྫ��acc
%���룺
%    w:cxd,���Է�����
%    testdata��nxd ����������������
%    testlabel��nx1 �������ݱ�ǩ
%�����
%    acc���ڲ��Լ��ϵķ���׼ȷ��
dim=size(w,2);
y=w*testdata(:,1:dim)';%����ѡ��ע�����ݽ��٣������������ά�������ģ��û�н��и��£��������յ�ģ��w_t��ά�����������ά�Ȳ�һ�¡�
[probablity ,pred] = max(y);
acc = mean(testlabel(:) == pred(:));