function [edata,efeature,edgeStruct]=node2edge(data,labl,feature,pccparam,ffparam)
% Transforming node data to edge data (de novo)
% data: m by n matrix of node data, where m is the number
%       of node features, n is the number of samples
% labl: 1,2,3,...,K; labeling the sample classes
% feature: m by 1 cell array of strings, feature names
% pccparam: double; criterion for selecting differentially 
%           correlated feature pairs.(default 0.5) 
%           For a certain feature pair, there will be K PCCs 
%           with respect to K classes. Among these PCCs, if 
%           there are two PCCs between which the absolute 
%           difference is larger than the pccparam, then 
%           select the feature pair
% ffparam: double; criterion for filtering differential 
%          edge features.(default 0.5) 
%          For a certain edge feature, there will be K(K-1)/2
%          pvalues (one-against-one ttest) among K classes of 
%          samples. If all of the pvalues are smaller than 
%          ffparam, then select the edge feature
% edata: l by n matrix of edge data, where l is the number
%        of edge features, n is the number of samples
% efeature: l by 1 cell array of strings; edge feature names
%           taking the form 
%           "<name of node1>~<name of node2>~<class i>"
%           where name of node1 is prior to name of node2 
%           in lexicographic order
% edgeStruct: structured variable with fields:
%             "edge": l by 3 cell array; column 1 and 2 are 
%                     strings of nodes for each edge; 
%                     column 3 are labels of class whose 
%                     means and standard deviations are 
%                     used to transform the data
%             "feature": m' by 1 cell array of node features
%                        that is involved in the "edge"
%             "mu": m' by K matrix of mean values
%             "usd": m' by K matrix of uncorrected standard 
%                    deviation
% Created by Wanwei Zhang (wanway@hotmail.com)

if nargin<5
    ffparam=0.05;
end
if nargin<4
    pccparam=0.5;
end

[level,~,idx]=unique(labl);
nc=numel(level);
ns=size(data,2);
nf=size(data,1);

% generalized z-score transformation
zdata=zeros(nc*nf,ns);
mu=zeros(nf,nc);
usd=zeros(nf,nc);
for i=1:nc
    mu(:,i)=mean(data(:,labl==level(i)),2);
    usd(:,i)=std(data(:,labl==level(i)),1,2);
    zdata((1:nf)+(i-1)*nf,:)=(data-repmat(mu(:,i),1,ns))./...
        repmat(usd(:,i),1,ns);
end

% calculate of PCCs of each feature pair
pcc=zeros(nf,nf,nc);
for i=1:nc
    pcc(:,:,i)=corrcoef(data(:,idx==i)');
end
% absolute difference of PCCs
adpcc=zeros(nf,nf,nc*(nc-1)/2);
l=0;
for i=1:nc-1
    for j=i+1:nc
        l=l+1;
        adpcc(:,:,l)=abs(pcc(:,:,i)-pcc(:,:,j));
    end
end

idx=tril(any(adpcc>pccparam,3),-1);
edgeIdx_tmp=zeros(sum(idx(:)),2);
[edgeIdx_tmp(:,1),edgeIdx_tmp(:,2)]=find(idx);
edge_tmp=cell(size(edgeIdx_tmp));
edge_tmp(:)=feature(edgeIdx_tmp);

ne=size(edgeIdx_tmp,1);
edgeIdx=zeros(nc*ne,2);
edge=cell(nc*ne,3);
for i=1:nc
    edgeIdx((1:ne)+(i-1)*ne,:)=edgeIdx_tmp+(i-1)*nf;
    edge((1:ne)+(i-1)*ne,:)=[edge_tmp,num2cell(i*ones(ne,1))];
end

edata=zdata(edgeIdx(:,1),:).*zdata(edgeIdx(:,2),:);

pval=zeros(size(edata,1),nc*(nc-1)/2);
l=0;
for i=1:nc-1
    for j=i+1:nc
        l=l+1;
        pval(:,l)=mattest(edata(:,labl==level(i)),edata(:,labl==level(j)));
    end
end
idx=all(pval<ffparam,2);

edata=edata(idx,:);
edge=edge(idx,:);

% sort nodes of each edge and get efeature
efeature=cell(size(edge,1),1);
for i=1:size(edge,1)
    edge(i,1:2)=sort(edge(i,1:2));
    efeature{i}=[edge{i,1},'~',edge{i,2},'~',num2str(edge{i,3})];
end

edgeStruct.edge=edge;
temp=unique(edge(:,1:2));
idx=zeros(size(temp));
for i=1:numel(temp)
    idx(i)=find(strcmp(temp{i},feature));
end
edgeStruct.feature=feature(idx);
edgeStruct.mu=mu(idx,:);
edgeStruct.usd=usd(idx,:);
