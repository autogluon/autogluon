function Out = DeepInsight_train_norm(dset,NORM,Parm)
% Out = DeepInsight_train_norm(dset,NORM,Parm)

if exist('Parm')==0
   Parm.Method = ['tSNE']; % other methods 2) kpca or 3) pca 
   Parm.Max_Px_Size = 120;
   Parm.MPS_Fix = 1; % if this val is 1 then screen will be Max_Px_Size x Max_Px_Size (eg 120x120 )
                     % otherwise automatically decided by the distribution
                     % of the input data.
   Parm.ValidRatio = 0.1; % 0.1 of Train data will be used as Validation data
   Parm.Seed = 108; %random seed
end 

%Take data on "as is" basis; i.e., no change in Train and Test data
TrueLabel=[];
for j=1:dset.class
    TrueLabel=[TrueLabel,ones(1,dset.num_tr(j))*j];
end

%YTest=[];
%for j=1:dset.class
%    YTest=[YTest,ones(1,dset.num_tst(j))*j];
%end
%YTest=categorical(YTest)';
YTrain=categorical(TrueLabel)';


q=1:length(TrueLabel);
clear idx
for j=1:dset.class
    rng=q(double(TrueLabel)==j);
    rand('seed',Parm.Seed);
    idx{j} = rng(randperm(length(rng),round(length(rng)*Parm.ValidRatio)));
end
idx=cell2mat(idx);
dset.XValidation = dset.Xtrain(:,idx);
dset.Xtrain(:,idx) = [];
YValidation = YTrain(idx);
YTrain(idx) = [];


switch NORM
    case 1
        % Norm-3 in org code
        Out.Norm=1;
    %fprintf(Parm.fid,'\nNORM-1\n');
    fprintf('\nNORM-1\n');
    %########### Norm-1 ###################
    Out.Max=max(dset.Xtrain')';
    Out.Min=min(dset.Xtrain')';
    dset.Xtrain=(dset.Xtrain-Out.Min)./(Out.Max-Out.Min);
    dset.XValidation = (dset.XValidation-Out.Min)./(Out.Max-Out.Min);
    %dset.Xtest = (dset.Xtest-Out.Min)./(Out.Max-Out.Min);
    dset.Xtrain(isnan(dset.Xtrain))=0;
    %dset.Xtest(isnan(dset.Xtest))=0;
    dset.XValidation(isnan(dset.XValidation))=0;
    dset.XValidation(dset.XValidation>1)=1;
    dset.XValidation(dset.XValidation<0)=0;
    %dset.Xtest(dset.Xtest>1)=1;
    %dset.Xtest(dset.Xtest<0)=0;
    %######################################
    
    case 2
        % norm-6 in org ocde
        Out.Norm=2;
    %fprintf(Parm.fid,'\nNORM-2\n');
    fprintf('\nNORM-2\n');
    %########### Norm-2 ###################
    Out.Min=min(dset.Xtrain')';
    dset.Xtrain=log(dset.Xtrain+abs(Out.Min)+1);
    indV = dset.XValidation<Out.Min;
    %Out.indT = dset.Xtest<Out.Min;
    for j=1:size(dset.Xtrain,1)
        dset.XValidation(j,indV(j,:))=Out.Min(j);
        %dset.Xtest(j,Out.indT(j,:))=Out.Min(j);
    end
    dset.XValidation = log(dset.XValidation+abs(Out.Min)+1);
    %dset.Xtest=log(dset.Xtest+abs(Out.Min)+1);
    Out.Max=max(max(dset.Xtrain));
    dset.Xtrain=dset.Xtrain/Out.Max;
    dset.XValidation = dset.XValidation/Out.Max;
    %dset.Xtest = dset.Xtest/Out.Max;
    dset.XValidation(dset.XValidation>1)=1;
    %dset.Xtest(dset.Xtest>1)=1;
    %######################################
end


Q.data = dset.Xtrain;
Q.Method = Parm.Method;%['tSNE'];
Q.Max_Px_Size = Parm.Max_Px_Size;%120;

if Parm.MPS_Fix==1
    [Out.M,Out.xp,Out.yp,Out.A,Out.B,Out.Base] = Cart2Pixel(Q,Q.Max_Px_Size,Q.Max_Px_Size);
else
    [Out.M,Out.xp,Out.yp,Out.A,Out.B,Out.Base] = Cart2Pixel(Q);
end
fprintf('\n Pixels: %d x %d\n',Out.A,Out.B);
clear Q
dset.Xtrain=[];
close all;
%for j=1:length(YTest)
%    XTest(:,:,1,j) = ConvPixel(dset.Xtest(:,j),xp,yp,A,B,Base,0);
%end
%dset.Xtest=[];

for j=1:length(YValidation)
    XValidation(:,:,1,j) = ConvPixel(dset.XValidation(:,j),Out.xp,Out.yp,Out.A,Out.B,Out.Base,0);
end
dset.XValidation=[];
for j=1:length(YTrain)
    XTrain(:,:,1,j) = Out.M{j};
end
clear M X Y
Out = rmfield(Out,'M');

% change parameters as desired
optimVars = [
    optimizableVariable('filterSize',[2 10],'Type','integer')
    optimizableVariable('filterSize2',[4 30],'Type','integer')
    optimizableVariable('initialNumFilters',[2 16],'Type','integer')
    optimizableVariable('InitialLearnRate',[1e-5 1e-1],'Transform','log')
    optimizableVariable('Momentum',[0.8 0.95])
    optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')];

ObjFcn = makeObjFcn2(XTrain,YTrain,XValidation,YValidation); % working-model

current_dir=pwd;
cd DeepResults
BayesObject = bayesopt(ObjFcn,optimVars,...
    'MaxObj',100,...
    'MaxTime',24*60*60,...
    'IsObjectiveDeterministic',false,...    
    'UseParallel',false);


Out.bestIdx = BayesObject.IndexOfMinimumTrace(end);
Out.fileName = BayesObject.UserDataTrace{Out.bestIdx};
savedStruct = load(Out.fileName);
Out.valError = savedStruct.valError
cd(current_dir);

%[YPredicted,probs] = classify(savedStruct.trainedNet,XTest);
%testError = 1 - mean(YPredicted == YTest)
%Accuracy = 1-testError
