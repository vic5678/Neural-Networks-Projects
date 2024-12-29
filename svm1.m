% Load CIFAR-10 Dataset
clc; clear; close all;
trainData = [];
trainLabels = [];

% Assuming CIFAR-10 data is already loaded as trainData, trainLabels, testData, testLabels

% [trainData, trainLabels, testData, testLabels] = loadCIFAR10();
batch = load('data_batch_1.mat');   % Load only the first batch for simplicity
trainData = double(batch.data(1:10000, :)) / 255.0;  % Normalize pixel values to [0, 1]
trainLabels = batch.labels(1:10000);                 % First 1000 labels
trainData = (trainData - mean(trainData)) ./ std(trainData); % Standardization

%}
%{
num_Batches=5;
for i = 1:num_Batches
   batch = load(sprintf('data_batch_%d.mat', i)); % Load each batch
   trainData = [trainData; double(batch.data)/255.0];   % Concatenate data
   trainLabels = [trainLabels; batch.labels];     % Concatenate labels
end
%}

batch1=load('test_batch.mat');
testData = double(batch1.data(1:10000, :)) / 255.0;  % Normalize pixel values to [0, 1]
testLabels = batch1.labels(1:10000);                 % First 1000 labels

% Convert the dataset to two classes (e.g., Class 0 and Class 1)
selected_classes = [6, 9]; % Choose two classes for binary classification

% Extract only selected classes from training data
binary_train_idx = ismember(trainLabels, selected_classes);
binary_test_idx = ismember(testLabels, selected_classes);

trainData = trainData(binary_train_idx, :);
trainLabels = trainLabels(binary_train_idx);

testData = testData(binary_test_idx, :);
testLabels = testLabels(binary_test_idx);
testData = (testData - mean(trainData)) ./ std(trainData); % Standardization

% Convert labels to binary (-1, 1) for SVM compatibility
trainLabels(trainLabels == selected_classes(1)) = -1;
trainLabels(trainLabels == selected_classes(2)) = 1;
testLabels(testLabels == selected_classes(1)) = -1;
testLabels(testLabels == selected_classes(2)) = 1;

% Train SVM
SVMModel = fitcsvm(trainData, trainLabels, ...
   'KernelFunction', 'linear', 'BoxConstraint', 0.01);
%SVMModel = fitcsvm(trainData, trainLabels,  'KernelFunction', 'polynomial', 'PolynomialOrder', 4, 'BoxConstraint', 0.01);
%SVMModel = fitcsvm(trainData, trainLabels, 'KernelFunction', 'rbf', 'KernelScale', 'auto', 'BoxConstraint', 1);

 %Predict on train data
predictions = predict(SVMModel, trainData);

% Calculate accuracy
train_accuracy = mean(predictions == trainLabels) * 100;
fprintf('Train Accuracy: %.2f%%\n', train_accuracy);
% Predict on test data
predictions = predict(SVMModel, testData);

% Calculate accuracy
test_accuracy = mean(predictions == testLabels) * 100;

% Display results
fprintf('Test Accuracy: %.2f%%\n', test_accuracy);
