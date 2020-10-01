%   X: data matrix, each row is one observation, each column is one feature
%   d: reduced dimension
%   Y: dimensionanlity-reduced data
%   Warning: This function is not optimized for very high dimensional data!

%   Copyright by Quan Wang, 2011/05/10
%   Please cite: Quan Wang. Kernel Principal Component Analysis and its 
%   Applications in Face Recognition and Active Shape Models. 
%   arXiv:1207.3538 [cs.CV], 2012. 

function [Y, eigVector, eigValue]=PCA(X,d)

%% eigenvalue analysis
Sx=cov(X);
[V,D]=eig(Sx);
eigValue=diag(D);
[eigValue,IX]=sort(eigValue,'descend');
eigVector=V(:,IX);

%% normailization
norm_eigVector=sqrt(sum(eigVector.^2));
eigVector=eigVector./repmat(norm_eigVector,size(eigVector,1),1);

%% dimensionality reduction
eigVector=eigVector(:,1:d);
Y=X*eigVector;

