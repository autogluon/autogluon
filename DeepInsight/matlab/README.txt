DeepInsight

1) Main.m is the main program to execute DeepInsight algorithm.

2) Ringnorm-delve dataset is provided to verify the code.

3) MaxTime is the maximum training time for the DeepInsight algorithm, which is currently set as 24 hours per norm (note 2 norms are used, so maxim training time is 48 hours). The MaxTime is set in DeepInsight_train_norm.m code at Line 132. Please reduce this time to verify the algorithm. If you prefer 2 hours then set MaxTime as 2*60*60 at Line 132.

4) Change the Convolutional Neural Network (CNN) architecture as desired.

5) The MaxEpochs is set to 100 in makeObjFcn2.m for Bayesian Optimization at Line 140. Please change this as desired. This will affect the processing time of training.



Contact
Email: alok.fj@gmail.com

Reference
Sharma et al., DeepInsight: a methodology to transform a non-image data to an image for convolution neural netwrok architecture, Scientific Report, 2019.


