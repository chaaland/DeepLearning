%% CS294A/CS294W Self-taught Learning Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  self-taught learning. You will need to complete code in feedForwardAutoencoder.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises.
%
%% ======================================================================
%  STEP 0: Here we provide the relevant parameters values that will
%  allow your RICA to get good filters; you do not need to 
%  change the parameters below.
addpath(genpath('..'))
imgSize = 28;
global params;
params.patchWidth=9;           % width of a patch
params.n=params.patchWidth^2;   % dimensionality of input to RICA
params.lambda = 0.0005;   % sparsity cost
params.numFeatures = 32; % number of filter banks to learn
params.epsilon = 1e-2;   

SAMPLE_PATCHES = false;
LOAD_WEIGHTS = true;
DISPLAY_NETWORK = false;

%% ======================================================================
%  STEP 1: Load data from the MNIST database
%
%  This loads our training and test data from the MNIST database files.
%  We have sorted the data for you in this so that you will not have to
%  change it.

% Load MNIST database files
mnistData   = loadMNISTImages('../common/train-images-idx3-ubyte');
mnistLabels = loadMNISTLabels('../common/train-labels-idx1-ubyte');

numExamples = size(mnistData, 2);
% 50000 of the data are pretended to be unlabelled
unlabeledSet = 1:50000;
unlabeledData = mnistData(:, unlabeledSet);

% the rest are equally split into labelled train and test data
trainSet = 50001:55000;
testSet = 55001:60000;
trainData   = mnistData(:, trainSet);
trainLabels = mnistLabels(trainSet)' + 1; % Shift Labels to the Range 1-10
% only keep digits 0-4, so that unlabelled dataset has different distribution
% than the labelled one.
removeSet = find(trainLabels > 5);
trainData(:,removeSet)= [] ;
trainLabels(removeSet) = [];

testData   = mnistData(:, testSet);
testLabels = mnistLabels(testSet)' + 1;   % Shift Labels to the Range 1-10
% only keep digits 0-4
removeSet = find(testLabels > 5);
testData(:,removeSet)= [] ;
testLabels(removeSet) = [];


% Output Some Statistics
fprintf('# examples in unlabeled set: %d\n', size(unlabeledData, 2));
fprintf('# examples in supervised training set: %d\n\n', size(trainData, 2));
fprintf('# examples in supervised testing set: %d\n\n', size(testData, 2));
fflush(stdout);

%% ======================================================================
%  STEP 2: Train the RICA
%  This trains the RICA on the unlabeled training images. 

%  Randomly initialize the parameters
randTheta = randn(params.numFeatures,params.n)*0.01;  % 1/sqrt(params.n);
randTheta = randTheta ./ repmat(sqrt(sum(randTheta.^2,2)), 1, size(randTheta,2)); 
randTheta = randTheta(:);

% subsample random patches from the unlabelled+training data
if(SAMPLE_PATCHES)
patches = samplePatches([unlabeledData,trainData],params.patchWidth,200000);
save("patches.mat", "patches");
else
load "patches.mat";
end

%configure minFunc
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 1000;
if(LOAD_WEIGHTS)
load 'opttheta.mat';
else
%  Find opttheta by running the RICA on all the training patches.
%  You will need to whitened the patches with the zca2 function 
%  then call minFunc with the softICACost function as seen in the RICA exercise.
[whitened_patches, V] = multiple_returns_zca2(patches);
[opttheta, cost, exitflag] = minFunc( @(theta) softICACost(theta, whitened_patches, params), randTheta, options); 
save('opttheta.mat', "opttheta", "V");
end

% reshape visualize weights
W = reshape(opttheta, params.numFeatures, params.n);

if(DISPLAY_NETWORK)
display_network(W');
print -djpeg weights.jpg   % save the visualization to a file 
end

%% ======================================================================

%% STEP 3: Extract Features from the Supervised Dataset
% pre-multiply the weights with whitening matrix, equivalent to whitening
% each image patch before applying convolution. V should be the same V
% returned by the zca2 when you whiten the patches.

W = W*V;
%  reshape RICA weights to be convolutional weights.
W = reshape(W, params.numFeatures, params.patchWidth, params.patchWidth);
W = permute(W, [2,3,1]);

%  setting up convolutional feed-forward. You do need to modify this code.
filterDim = params.patchWidth;
poolDim = 5;
numFilters = params.numFeatures;
trainImages=reshape(trainData, imgSize, imgSize, size(trainData, 2));
testImages=reshape(testData, imgSize, imgSize, size(testData, 2));

%  Compute convolutional responses
faster_trainAct = faster_feedforwardRICA(filterDim, poolDim, numFilters, trainImages, W);
faster_testAct = faster_feedforwardRICA(filterDim, poolDim, numFilters, testImages, W);

%  reshape the responses into feature vectors
featureSize = size(faster_trainAct,1)*size(faster_trainAct,2)*size(faster_trainAct,3);
trainFeatures = reshape(faster_trainAct, featureSize, size(trainData, 2));
testFeatures = reshape(faster_testAct, featureSize, size(testData, 2));

%% ======================================================================
%% STEP 4: Train the softmax classifier

numClasses  = 5; % doing 5-class digit recognition
% initialize softmax weights randomly
randTheta2 = randn(numClasses-1, featureSize)*0.01;  % 1/sqrt(params.n);
randTheta2 = randTheta2 ./ repmat(sqrt(sum(randTheta2.^2,2)), 1, size(randTheta2,2)); 
randTheta2 = randTheta2';
randTheta2 = randTheta2(:);

%  Use minFunc and softmax_regression_vec from the previous exercise to 
%  train a multi-class classifier. 
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 300;

% optimize
opttheta2 = minFunc(@softmax_regression_vec, randTheta2, options, trainFeatures, trainLabels);
opttheta2 = reshape(opttheta2, featureSize, []);
opttheta2=[opttheta2, zeros(featureSize,1)]; % expand theta to include the last class.
save("opttheta2.mat", "opttheta2");

%%======================================================================
%% STEP 5: Testing 
% Compute Predictions on train and test sets using softmaxPredict
% and softmaxModel

[max_val, train_pred] = max(opttheta2.' * trainFeatures);
[max_val, pred] = max(opttheta2.' * testFeatures);

multi_classifier_accuracy(opttheta2, trainFeatures, trainLabels)
multi_classifier_accuracy(opttheta2, testFeatures, testLabels)

% Classification Score
fprintf('Train Accuracy: %f%%\n', 100*mean(train_pred(:) == trainLabels(:)));
fprintf('Test Accuracy: %f%%\n', 100*mean(pred(:) == testLabels(:)));
% You should get 100% train accuracy and ~99% test accuracy. With random
% convolutional weights we get 97.5% test accuracy. Actual results may
% vary as a result of random initializations

