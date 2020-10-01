%   X: data matrix, each row is one observation, each column is one feature
%   D: pair-wise distance matrix

%   Copyright by Quan Wang, 2011/05/10
%   Please cite: Quan Wang. Kernel Principal Component Analysis and its 
%   Applications in Face Recognition and Active Shape Models. 
%   arXiv:1207.3538 [cs.CV], 2012. 

function D=distanceMatrix(X)

N=size(X,1);

XX=sum(X.*X,2);
XX1=repmat(XX,1,N);
XX2=repmat(XX',N,1);

D=XX1+XX2-2*(X*X');
D(D<0)=0;
D=sqrt(D);