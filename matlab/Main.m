clear all;
close all;

%Description of Datasets
% 1. dataset1.mat is RNA-seq (1.5GB)
% 2. dataset2.mat is ringnorm-DELVE (560KB)
% 3. dataset3.mat is vowels (3.6MB)
% 4. dataset4.mat is Madelon (2.1MB)
% 5. dataset5.mat is Relathe (696KB)

Parm.fid=fopen('Results.txt','a+');
Parm.Method = ['tSNE']; % other methods 2) kpca or 3) pca 
Parm.Max_Px_Size = 120;
Parm.MPS_Fix = 1; % if this val is 1 then screen will be 
                  % Max_Px_Size x Max_Px_Size (e.g. 120x120), otherwise 
                  % automatically decided by the distribution of the input data.
Parm.ValidRatio = 0.1; % ratio of validation data/Training data
Parm.Seed = 108; % random seed to distribute training and validation sets

for j=2:2%1:3 %1:5
j
current_dir=pwd;
cd Data
load(['dataset',num2str(j),'.mat']);
cd(current_dir);

model = DeepInsight_train(dset,Parm);
accuracy = DeepInsight_test(dset,model);
fprintf(Parm.fid,'\nDeepInsight accuracy %6.2f\n',accuracy);
end

fclose(Parm.fid);
