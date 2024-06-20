clear;
load('handwritten.mat');
Y=Y+1;
N=length(Y);

% for j=1:10
% 
%     idx=randperm(N);
%     tabu=tabulate(Y);
%     classCount=zeros(max(tabu(:,2)),length(tabu(:,1)));
%     countId=zeros(1,length(tabu(:,1)));
%     for i=1:N
%         idxClass=Y(idx(i));
%         countId(idxClass)=countId(idxClass)+1;
%         classCount(countId(idxClass),idxClass)=idx(i);
%     end
%         
%     idxtrain=classCount(1:5,:);
%     idxval=classCount(6:8,:);
%     idxtest=classCount(9:end,:);
%     train_idx1=idxtrain(:);
%     valid_idx1=idxval(:);
%     test_idx1=idxtest(:);
%     train_idx0=train_idx1-1;
%     test_idx0=test_idx1-1;
%     valid_idx0=valid_idx1-1;
%     
%     save("handtest_idx0"+num2str(j),'test_idx0');
%     save("handtest_idx1"+num2str(j),'test_idx1');
%     save("handtrain_idx0"+num2str(j),'train_idx0');
%     save("handtrain_idx1"+num2str(j),'train_idx1');
%     save("handvalid_idx0"+num2str(j),'valid_idx0');
%     save("handvalid_idx1"+num2str(j),'valid_idx1');
% end
f1=X{1};
feat1=normalize(f1,1,"norm",2);
feat1=sparse(feat1);
save hand1feat1 feat1;
f2=X{2};
feat2=normalize(f2,1,"norm",2);
feat1=sparse(feat2);
save hand2feat1 feat1;
f3=X{3};
feat3=normalize(f3,1,"norm",2);
feat1=sparse(feat3);
save hand3feat1 feat1;
f4=X{4};
feat4=normalize(f4,1,"norm",2);
feat1=sparse(feat4);
save hand4feat1 feat1;
f5=X{5};
feat5=normalize(f5,1,"norm",2);
feat1=sparse(feat5);
save hand5feat1 feat1;
f6=X{6};
feat6=normalize(f6,1,"norm",2);
feat1=sparse(feat6);
save hand6feat1 feat1;
f=[X{1},X{2},X{3},X{4},X{5},X{6}];
feat=normalize(f,1,"norm",2);
feat1=sparse(feat);
save handfeat1 feat1;