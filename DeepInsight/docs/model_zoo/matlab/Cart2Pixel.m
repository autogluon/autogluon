function [M,xp,yp,A,B,Base] = Cart2Pixel(Q,A,B)
% [M,xp,yp,A,B,Base] = Cart2Pixel(Q,A,B)
%
% Q.data should be in no_of_genes x no_of_samples format
% 
% Q.Method
% n=10;
% x = randn(n,1);
% y = randn(n,1);
% Method = 1) kpca for kernel pca 
%       or 2) tSNE for t-Distributed Stochastic Neighbor Embedding
%       or 3) pca for principal component analysis
%
% Q.Max_Px_Size is max(A,B)

if any(strcmp('data',fieldnames(Q)))~=1
    disp('no data provided')
end
if any(strcmp('Method',fieldnames(Q)))~=1
    Q.Method=['tSNE'];
end
if any(strcmp('Max_Px_Size',fieldnames(Q)))~=1
    Q.Max_Px_Size=30;
end
if any(strcmp('Dist',fieldnames(Q)))~=1
    Q.Dist='cosine';
end
if exist('A')==1
    A=A-1;
end
if exist('B')==1
    B=B-1;
end



if strcmp(lower(Q.Method),'kpca')==1
    disp('kpca is used');
    DIST=distanceMatrix(Q.data);
    DIST(DIST==0)=inf;
    DIST=min(DIST);
    para=5*mean(DIST);
    [Y, ~]=kPCA(Q.data,2,'gaussian',para);
elseif strcmp(lower(Q.Method),'pca')==1
    disp('pca is used');
    Y=PCA(Q.data,2);
else
    if size(Q.data,1)<5000
        disp('tSNE with exact algorithm is used');
        Y=tsne(Q.data,'Algorithm','exact','Distance',Q.Dist);
    else
        disp('tSNE with burneshut algorithm is used');
        Y=tsne(Q.data,'Algorithm','barneshut','Distance',Q.Dist);
    end
end
x=Y(:,1);
y=Y(:,2);
[n,no_samples]=size(Q.data);
clear Y DIST para


% should have a nearly square bounding rectangle
[xrect,yrect] = minboundrect(x,y);

figure
hold on;
plot(xrect,yrect,'k-');
plot(x,y,'o');

%gradient (m) of a line y=mx+c
grad = (yrect(2)-yrect(1))/(xrect(2)-xrect(1));
theta = atan(grad);

%Rotation matrix
%theta=180-theta
R=[cos(theta) sin(theta);-sin(theta) cos(theta)];

% rotated rectangle
zrect = R*[xrect';yrect'];

% rotated data
z = R*[x';y'];

plot(z(1,:),z(2,:),'o');
plot(zrect(1,:),zrect(2,:),'r-');
axis square

% Find nearest points
%tic
min_dist = Inf;
min_p1 = 0;
min_p2 = 0;
for p1 = 1:n
    for p2 = p1+1:n
        d = (z(1,p1)-z(1,p2))^2+(z(2,p1)-z(2,p2))^2;
        if d < min_dist && p1 ~= p2 && d>0
            min_p1 = p1;
            min_p2 = p2;
            min_dist = d;
        end
    end
end
%Time=toc
plot([z(1,min_p1),z(1,min_p2)],[z(2,min_p1),z(2,min_p2)],'k.');

% Find distance between two nearest points
dmin = norm(z(:,min_p1)-z(:,min_p2));

% Find coordinates of pixel frame (A,B)
rec_x_axis = abs(zrect(1,1)-zrect(1,2));
rec_y_axis = abs(zrect(2,2)-zrect(2,3));

% if dmin is sqrt(2)del, then what is A and B in terms of del (where del is
% one pixel length)
if exist('A')==0 & exist('B')==0
Precision_old=sqrt(2);
A = ceil(rec_x_axis*Precision_old/dmin);
B = ceil(rec_y_axis*Precision_old/dmin);
%Max_Px_Size = 50;%300;
if max([A,B]) > Q.Max_Px_Size
    Precision = Precision_old*Q.Max_Px_Size/max([A,B]);
    A = ceil(rec_x_axis*Precision/dmin);
    B = ceil(rec_y_axis*Precision/dmin);
end
end
%A=25; B=25;

% Transform from cartesian coordinates to pixels
xp = round(1+(A*(z(1,:)-min(z(1,:)))/(max(z(1,:))-min(z(1,:)))));
yp = round(1+(-B)*(z(2,:)-max(z(2,:)))/(max(z(2,:))-min(z(2,:))));
A=max(xp);
B=max(yp);

Base=1;%min(min(Q.data)) - (max(max(Q.data))-min(min(Q.data)))*0.03;%mean(mean(Q.data));%
FIG=0; % if FIG=1 then plots will appear
%S=zeros(A,B);
for j=1:no_samples
%     if j==1| j==no_samples
%        FIG=1;
%     else
%        FIG=0;
%     end
    M{j} = ConvPixel(Q.data(:,j),xp,yp,A,B,Base,FIG);
    %S=S+M{j};
end
% S=S/no_samples;
% for j=1:no_samples
%     M{j}=M{j}-S;
% end
