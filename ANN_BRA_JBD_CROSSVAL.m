clear; close all; clc;
filename = 'Database source'; %specify path of your database file
z = xlsread(filename);
%rng(seed) %set integer value to freeze the results
n = randperm(1022);
input = z(n(1:1022),1:5);
target = z(n(1:1022),6);

%Input and output sets
format longg 
x = input';
t = target';

%Algorithm
trainFcn = 'trainbr';  %Bayesian regularization

%Neural network architecture
hiddenLayerSize1 = 8;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%

%K-FOLD CROSS-VALIDATION
netcv = fitnet([hiddenLayerSize1], trainFcn);
netcv.performFcn='mse';
netcv.trainParam.epochs = 1000;
netcv.trainParam.goal = 0;
netcv.trainParam.max_fail = 1.00e+010;
netcv.trainParam.min_grad = 1.00e-010;
netcv.trainParam.mu = 0.0010; %mu
netcv.trainParam.mu_dec = 0.0010; %mu_dec
netcv.trainParam.mu_inc = 10; %mu_inc
netcv.trainParam.mu_max = 1.00e+010;
netcv.trainParam.show = 5;
netcv.trainParam.time = inf;

%Activation functions
netcv.layers{1}.transferFcn = 'tansig';
netcv.layers{2}.transferFcn = 'purelin';

k=5; %specify number of cross-validation (k)
for i=1:k
    c = cvpartition(length(t),'KFold',k);
    trainingIdx = training(c,i); 
    testIdx = test(c,i); 
    sum(trainingIdx==1);
    sum(testIdx==1);
    XTrain = input(trainingIdx,:)'; 
    YTrain = target(trainingIdx)';
    
    %Data normalization
    [XTrainN,xscv] = mapminmax(XTrain); 
    [YTrainN,tscv] = mapminmax(YTrain); 
    XTest = input(testIdx,:);   
    YTest = target(testIdx);   

    %Data division strategy
    [traincvInd,valcvInd,testcvInd] = dividerand(length(YTrain),1.00,0,0); 
    netcv.divideParam.trainRatio = 100/100;
    netcv.divideParam.valRatio = 0/100;
    netcv.divideParam.testRatio = 0/100;

    XTestN = mapminmax('apply',XTest',xscv);
    YTestN = mapminmax('apply',YTest',tscv);

    %Train the Network
    [netcv,trcv] = train(netcv,XTrainN,YTrainN);

    anTRAINcv = netcv(XTrainN);
    aTRAINcv = mapminmax('reverse',anTRAINcv,tscv)';
    anTESTcv = netcv(XTestN);
    aTESTcv = mapminmax('reverse',anTESTcv,tscv);
    OutputCVtrainANN = aTRAINcv(trcv.trainInd)';
    OutputCVtrainVALANN = aTRAINcv(trcv.testInd)';
    ScaledoutputCVtestANN = netcv(XTestN); 
    OriginaloutputCVtestANN = mapminmax('reverse',ScaledoutputCVtestANN,tscv);

    %Final performance indicators per each fold on train and validation/test set
    performanceCV_TRAIN = perform(netcv,aTRAINcv,YTrainN) 
    performanceCV_TEST = perform(netcv,anTESTcv,YTestN) 

    %Plot number of epochs
    epochs=trcv.num_epochs;
    bestepoch=trcv.best_epoch;
    
    %Plot number of efficient network parameters
    efficientparameters=trcv.gamk(bestepoch)
    
    %Total number of network parameters
    totalnumberofparameters = size(getwb(netcv))
    
    %Correlation and determination coefficients
    R_TEST = corrcoef(aTESTcv,YTest);
    R_TRAIN = corrcoef(aTRAINcv,YTrain);
    
end