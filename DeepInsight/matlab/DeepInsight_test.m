function Accuracy = DeepInsight_test(dset,Out)
% Accuracy = DeepInsight_test(dset,Out)

YTest=[];
for j=1:dset.class
   YTest=[YTest,ones(1,dset.num_tst(j))*j];
end
YTest=categorical(YTest)';
if Out.Norm==1
    fprintf('\nNORM-1\n');
    %########### Norm-1 ###################
    %Out.Max=max(dset.Xtrain')';
    %Out.Min=min(dset.Xtrain')';
    %dset.Xtrain=(dset.Xtrain-Out.Min)./(Out.Max-Out.Min);
    %dset.XValidation = (dset.XValidation-Out.Min)./(Out.Max-Out.Min);
    dset.Xtest = (dset.Xtest-Out.Min)./(Out.Max-Out.Min);
    %dset.Xtrain(isnan(dset.Xtrain))=0;
    dset.Xtest(isnan(dset.Xtest))=0;
    %dset.XValidation(isnan(dset.XValidation))=0;
    %dset.XValidation(dset.XValidation>1)=1;
    %dset.XValidation(dset.XValidation<0)=0;
    dset.Xtest(dset.Xtest>1)=1;
    dset.Xtest(dset.Xtest<0)=0;
    %######################################
else
    fprintf('\nNORM-2\n');
    %########### Norm-2 ###################
    %Out.Min=min(dset.Xtrain')';
    %dset.Xtrain=log(dset.Xtrain+abs(Out.Min)+1);
    %indV = dset.XValidation<Out.Min;
    Out.indT = dset.Xtest<Out.Min;
    for j=1:size(dset.Xtest,1)
    %    dset.XValidation(j,Out.indV(j,:))=Out.Min(j);
        dset.Xtest(j,Out.indT(j,:))=Out.Min(j);
    end
    %dset.XValidation = log(dset.XValidation+abs(Out.Min)+1);
    dset.Xtest=log(dset.Xtest+abs(Out.Min)+1);
    %Out.Max=max(max(dset.Xtrain));
    %dset.Xtrain=dset.Xtrain/Out.Max;
    %dset.XValidation = dset.XValidation/Out.Max;
    dset.Xtest = dset.Xtest/Out.Max;
    %dset.XValidation(dset.XValidation>1)=1;
    dset.Xtest(dset.Xtest>1)=1;
    %######################################
end
for j=1:length(YTest)
   XTest(:,:,1,j) = ConvPixel(dset.Xtest(:,j),Out.xp,Out.yp,Out.A,Out.B,Out.Base,0);
end
dset.Xtest=[];

%Out.bestIdx = BayesObject.IndexOfMinimumTrace(end);
%Out.fileName = BayesObject.UserDataTrace{Out.bestIdx};
current_dir=pwd;
cd DeepResults
savedStruct = load(Out.fileName);
%Out.valError = savedStruct.valError
cd(current_dir);

[YPredicted,probs] = classify(savedStruct.trainedNet,XTest);
testError = 1 - mean(YPredicted == YTest)
Accuracy = 1-testError
end
