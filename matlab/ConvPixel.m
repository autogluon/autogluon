function M = ConvPixel(FVec,xp,yp,A,B,Base,FIG);
% M=ConvPixel(FVec,xp,yp,A,B,Base,FIG)
if nargin==6
    FIG=0;
end

n=length(FVec);
% Plot in pixels
M=ones(A,B)*Base;%zeros(A,B);%(-50000)*ones(A,B);%zeros(A,B);%(-30000)*ones(A,B);%
for j=1:n
    M(xp(j),yp(j))=FVec(j);
end

% if (xp,yp) has some duplicates then value in M will be overwritten by the
% last (xp,yp) value used in the for loop. Therefore, it would be better to
% find any duplicates and use an average of these duplicate values in M.
zp=[xp;yp];
[c,ia,ic]=unique(zp','rows');
ic=ic';
%duplicate_zp = unique(zp(:,sum(ic==ic')>1)','rows')';%c(find(hist(ic)>1),:)';
Len_ic = 1:length(ic);
ic_pos = unique(ic(sum(ic==ic')>1));%Len_ic(hist(ic)>1);
for j=1:length(ic_pos)
    duplicate_pos = Len_ic(ic==ic_pos(j));
    f=0;
    for k=1:length(duplicate_pos)
        f=f+FVec(duplicate_pos(k));
        %M(xp(duplicate_pos(k)),yp(duplicate_pos(k))) = FVec(duplicate_pos(k))/length(duplicate_pos);
    end
    M(xp(duplicate_pos(1)),yp(duplicate_pos(1))) = f/length(duplicate_pos);
end

% M=mat2gray(M);

if FIG==1
    %figure; imshow(M');
    figure; imagesc(M');
end