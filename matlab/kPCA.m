%   X: data matrix, each row is one observation, each column is one feature
%   d: reduced dimension
%   type: type of kernel, can be 'simple', 'poly', or 'gaussian'
%   para: parameter for computing the 'poly' and 'gaussian' kernel, 
%       for 'simple' it will be ignored
%   Y: dimensionanlity-reduced data
%   eigVector: eigen-vector, will later be used for pre-image
%       reconstruction

%   Copyright by Quan Wang, 2011/05/10
%   Please cite: Quan Wang. Kernel Principal Component Analysis and its
%   Applications in Face Recognition and Active Shape Models.
%   arXiv:1207.3538 [cs.CV], 2012.

function [Y, eigVector, eigValue]=kPCA(X,d,type,para)

%% check input
if ( strcmp(type,'simple') || strcmp(type,'poly') || ...
        strcmp(type,'gaussian') ) == 0
    Y=[];
    eigVector=[];
    fprintf(['\nError: Kernel type ' type ' is not supported. \n']);
    return;
end

N=size(X,1);

%% kernel PCA
K0=kernel(X,type,para);
oneN=ones(N,N)/N;
K=K0-oneN*K0-K0*oneN+oneN*K0*oneN;

%% eigenvalue analysis
%[V,D]=eig(K/N);
[V,D]=svd(K/N,0);
r=rank(D);
D=D(1:r,1:r);
V=V(:,1:r);
eigValue=diag(D);
[~,IX]=sort(eigValue,'descend');
eigVector=V(:,IX);
eigValue=eigValue(IX);

%% normailization
norm_eigVector=sqrt(sum(eigVector.^2));
eigVector=eigVector./repmat(norm_eigVector,size(eigVector,1),1);

%% dimensionality reduction
eigVector=eigVector(:,1:d);
Y=K0*eigVector;

