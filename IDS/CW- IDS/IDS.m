clear
clc

load cw_dataset_2class.txt 
data = cw_dataset_2class;

%readtable('cw_dataset.txt')

%pre-process dataset
%index_nan = find(data == nan);
% fillmissing(data, 'constant', 0)
% data(isnan(data)) = 0; %to replace nan with 0
%check dataset is balanced - pretty much
tabulate(data(:,42))


%normalise data as columns are of different range
boxplot(data) %check if large number of outliers 
%probplot(data) %check probability of data
% huge outliers in 6th column?
data_norm = normalize(data, 'range');
%boxplot(data_norm)

%shuffle the dataset 
data_norm_shuf = data_norm(randperm(size(data_norm, 1)), :);
%split
trainRatio = 0.66;
valRatio = 0;
testRatio = 0.33;
[trainInd,valInd,testInd] = dividerand(size(data_norm, 1),trainRatio,valRatio,testRatio);
trainingDataset = data_norm_shuf(trainInd,:);
testingDataset = data_norm_shuf(testInd,:);
%tabulate(testingDataset(:, 42))
%tabulate(trainingDataset(:, 42))

%Create training data, training target, testing data and testing target. 
trainingData = trainingDataset(:,1:41)';
trainingTarget = trainingDataset(:,42)';

testingData = testingDataset(:,1:41)';
testingTarget = testingDataset(:,42)';

%Select Pattern Recognition Neural Network (patternnet)
net = patternnet(10, 'trainlm'); %Construct a pattern network with one hidden layer of size 10.
%learing rate
%net.trainParam.lr = 0.1;

% each input must be a vector, so you will have a matrix with as many columns Q as there are samples. Then the target should be a 1xQ. 
[net, tr] = train(net, trainingData, trainingTarget); %train the neural network using the dataset
 
view(net)
plotperform(tr); %plot performance graph

yhat  = net(testingData); %Estimate the targets using the trained network.
perf = perform(net, testingTarget, yhat) %Assess the performance of the trained network. The default performance function is mean squared error.
%classes = vec2ind(yhat); may be usefull for one hot

%multi output classifier








