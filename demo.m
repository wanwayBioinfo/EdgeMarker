% Created by Wanwei Zhang (wanway@hotmail.com)

%% generating node data
nf=100;%number of features
ns=50;%number of samples
data=randn(nf,ns);
labl=randi(2,ns,1);

% feature names
feature=cell(nf,1);
for i=1:nf
    feature{i}=['f',num2str(i)];
end

% sample names
sample=cell(ns,1);
for i=1:ns
    sample{i}=['s',num2str(i)];
end

%% transforming node data into edge space
% using default parameters
[edata,efeature,edgeStruct]=node2edge(data,labl,feature);

% self-defined parameters
ffparam=0.05;
pccparam=0.8;
[edata,efeature,edgeStruct]=node2edge(data,labl,feature,pccparam,ffparam);

%% selecting edge features by SFFS algorithm
% using default parameters
[ft_seq,J_seq]=sffs(edata,labl);

% self-defined parameters
sffsparam.svmparam='-t 2 -q';
sffsparam.maxIter=100;
sffsparam.kfold=10;
sffsparam.nrep=5;
sffsparam.tempsave='sffs_res.mat';
[ft_seq,J_seq]=sffs(edata,labl,sffsparam);

%% cross validation for selected edge feature sets
i_s=8;
edgeStruct_s=edgeStruct;
edgeStruct_s.edge=edgeStruct.edge(ft_seq{i_s},:);

pred=zeros(size(labl));
cvIdx=crossvalind('kfold',ns,10);% 10-fold CV
for i=1:max(cvIdx)
    edata_train=getEdgeData(edgeStruct_s,data(:,cvIdx~=i),feature);
    labl_train=labl(cvIdx~=i);
    edata_test=getEdgeData(edgeStruct_s,data(:,cvIdx==i),feature);
    labl_test=labl(cvIdx==i);
    
    svmobj=libsvmtrain(labl_train,edata_train','-t 0 -q');
    pred(cvIdx==i)=libsvmpredict(labl_test,edata_test',svmobj,'-q');
end

acc=100*mean(pred==labl);
