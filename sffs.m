function [ft_seq,J_seq]=sffs(data,labl,sffsparam)
% Sequential Forward Floating Selection algorithm
% data: m by n matrix of node data, where m is the number
%       of node features, n is the number of samples
% labl: 1,2,3,...,K; labeling the sample classes
% sffsparam: parameters for sffs algorithm; structured 
%            variable with fields:
%            "svmparam": parameter for libsvm, 
%                        default svmparam='-t 0 -q'
%            "maxIter": maximumn iterations of the 
%                       algorithm, 
%                       default maxIter=50
%            "kfold": kfold cross validation for evaluating 
%                     the goodness of feature sets, 
%                     default kfold=0 (leave-one-out CV)
%            "nrep": repeat times of cross valiations for 
%                    a given feature set, 
%                    default nrep=1.
%            "tempsave": file name to which the temporary 
%                        workspace is saved, 
%                        default tempsave='sffs_res.mat'. 
%                        tempsave='' suppress saving action
% ft_seq: cell vector of indices of selected feature sets
% J_seq: vector of cross validation accuracies of the 
%        selected feature sets
% Created by Wanwei Zhang (wanway@hotmail.com)

if nargin<3 || isempty(options)
    sffsparam.svmparam='-t 0 -q';
    sffsparam.maxIter=50;
    sffsparam.kfold=0;
    sffsparam.nrep=1;
    sffsparam.tempsave='sffs_res.mat';
end

if matlabpool('size')==0
    parpool;
end

svmparam=sffsparam.svmparam;
maxIter=sffsparam.maxIter;
kfold=sffsparam.kfold;
nrep=sffsparam.nrep;
tempsave=sffsparam.tempsave;

if kfold>size(data,2)||kfold==0
    kfold=size(data,2);
    nrep=1;
end

% optional: data normalization for each feature
data=zscore(data,0,2);

iter=0;

universe=(1:size(data,1))';
ft_seq=cell(size(data,1),1); % recording optimal feature sets, corresponding to increased number of features.
J_seq=zeros(size(data,1),1); % optimal scores of ft_seq
current=[]; % current feature set to be considered
% ceiling=1; % maximum feature set that has been considered
% main body
while iter<maxIter
    iter=iter+1;
    fprintf('iter=%d,forwarding...\n',iter);
    pool=setdiff(universe,current);
    if isempty(pool)
        break;
    end
    J_pool=zeros(numel(pool),1);
    parfor i_p=1:numel(pool)
        temp=svmCV(data([current;pool(i_p)],:),labl,kfold,nrep,svmparam);
        J_pool(i_p)=mean(temp);
    end
    [J_max,idx]=max(J_pool);
    if J_seq(numel(current)+1)<J_max
        ft_seq{numel(current)+1}=[current;pool(idx)];
        J_seq(numel(current)+1)=J_max;
        current=[current;pool(idx)];
    else
        current=ft_seq{numel(current)+1};
    end
    fprintf('#features=%d,J_seq=%f\n',numel(current),J_seq(numel(current)));
    
    % floating
    fprintf('iter=%d,floating...\n',iter);
    while numel(current)>2
        J_ft=zeros(numel(current),1);
        parfor i_f=1:numel(current)
            ft_tmp=current([1:i_f-1,i_f+1:end]);
            temp=svmCV(data(ft_tmp,:),labl,kfold,nrep,svmparam);
            J_ft(i_f)=mean(temp);
        end
        [J_max,idx]=max(J_ft);
        if J_seq(numel(current)-1)<J_max 
            J_seq(numel(current)-1)=J_max;
            ft_seq{numel(current)-1}=current([1:idx-1,idx+1:end]);
            current=current([1:idx-1,idx+1:end]);
            fprintf('#features=%d,J_seq=%f\n',numel(current),J_seq(numel(current)));
        else
            break;
        end
    end
    
    % output results every 10 iterations
    if ~isempty(tempsave)
        if rem(iter,10)==0
            save(tempsave);
        end
    end
end

ft_seq(J_seq==0)=[];
J_seq(J_seq==0)=[];
% figure;plot(J_seq)

function acc=svmCV(data,labl,kfold,nrep,svmparam)
% using svmTrain -v mode to do cross validation
acc=zeros(nrep,1);
for i=1:nrep
    idx=randperm(numel(labl));
    acc(i)=libsvmtrain(labl(idx),data(:,idx)',[svmparam,' -v ',num2str(kfold),]);
end
