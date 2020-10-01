function Out  = DeepInsight_train(dset,Parm)
% Out = DeepInsight_train(dset,Parm);
%
% dset is a data struct (dset.Xtrain, dset.XValidation,...)
%
% Out is the output in struct format

fprintf(Parm.fid,'\nDataset: %s\n',dset.Set);
fprintf('\nDataset: %s\n',dset.Set);

Out1 = DeepInsight_train_norm(dset,1,Parm);
fprintf('\nNorm-1 valError %2.4f\n',Out1.valError);
fprintf(Parm.fid,'\nNorm-1 valError %2.4f\n',Out1.valError);

Out2 = DeepInsight_train_norm(dset,2,Parm);
fprintf('\nNorm-2 valError %2.4f\n',Out2.valError);
fprintf(Parm.fid,'\nNorm-2 valError %2.4f\n',Out2.valError);
% select best one from Out1 and Out2
if Out1.valError < Out2.valError
    Out = Out1;
else
    Out = Out2;
end

fprintf(Parm.fid,'\nDeepInsight valErr: %6.4f\n',Out.valError);

end
