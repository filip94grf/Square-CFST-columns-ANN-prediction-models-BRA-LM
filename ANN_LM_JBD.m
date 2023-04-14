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
trainFcn = 'trainlm';  %Levenberg-Marquardt.

%Neural network architecture and hyperparameters
hiddenLayerSize1 = 12;
net = fitnet([hiddenLayerSize1], trainFcn);
net.performFcn='mse';
net.trainParam.epochs = 1000;
net.trainParam.goal = 1.00e-06;
net.trainParam.max_fail = 15;
net.trainParam.min_grad = 1.00e-010;
net.trainParam.mu = 0.1; %mu
net.trainParam.mu_dec = 0.01; %mu_dec
net.trainParam.mu_inc = 10; %mu_inc
net.trainParam.mu_max = 1.00e+10;
net.trainParam.show = 5;
net.trainParam.time = inf;

%Activation functions
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';

%Division strategy
[trainInd,valInd,testInd] = dividerand(1022,0.70,0.15,0.15);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;


x_train = x(:,trainInd); 
t_train = t(trainInd);

x_test = x(:,testInd); 
t_test = t(testInd); 

%Train the Network
[xn,xs] = mapminmax(x_train); 
[tn,ts] = mapminmax(t_train); 
[net,tr] = train(net,xn,tn);

%Test the Network
yn = net(xn); 
xn1=mapminmax(x);
y=net(xn1);
an1 = sim(net,x);
an = sim(net,xn);
a = mapminmax('reverse',an1,ts); 

%Error values
e = gsubtract(a,y); 
performanceALLDATA = perform(net,a,t);

% View the Network (optionally) - Uncomment these lines to enable various plots.
%view(net)

% Plots
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
%figure, plotfit(net,x,t)

Outputtraindataset = t(tr.trainInd)'; 
OutputtrainANN = a(tr.trainInd)'; 
InputtraindatasetandANN = x(tr.trainInd)';

Outputvalidationdataset = t(tr.valInd)'; 
OutputvalidationANN = a(tr.valInd)'; 
InputvalidationdatasetandANN = x(tr.valInd)';

Outputtestdataset = t(tr.testInd)'; 
OutputtestANN = a(tr.testInd)'; 
InputtestdatasetandANN = x(tr.testInd)';

Outputalldataset = t;
Outputalldatasetscaled = tn;
Inputall = x;
Inputallscaled = xn;

%Performance functions
performanceALLDATA = perform(net,a,t);
performanceTRAININGSET = perform(net,OutputtrainANN,Outputtraindataset);
performanceVALIDATIONSET = perform(net,OutputvalidationANN,Outputvalidationdataset); 
performanceTESTSET = perform(net,OutputtestANN,Outputtestdataset); 

w1 = net.IW{1}; %the input-to-hidden layer weights
w2 = net.LW{2}; %the hidden-to-output layer weights
b1 = net.b{1}; %the input-to-hidden layer bias
b2 = net.b{2}; %the hidden-to-output layer bias
