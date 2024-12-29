% Step 1: Load CIFAR-10 dataset
num_Batches = 5; % Number of training batches
trainData = []; % Initialize training data
trainLabels = []; % Initialize training labels
%{
batch = load('data_batch_1.mat');   % Load only the first batch for simplicity
trainData = double(batch.data(1:8000, :)) / 255.0;  % Normalize pixel values to [0, 1]
trainLabels = batch.labels(1:8000);                 % First 1000 labels
%}

% Load and concatenate all training batches
for i = 1:num_Batches
    batch = load(sprintf('data_batch_%d.mat', i)); % Load each batch
    trainData = [trainData; double(batch.data) / 255.0]; % Normalize pixel values and concatenate data
    trainLabels = [trainLabels; batch.labels]; % Concatenate labels
end
%}
% Load test data
batch1 = load('test_batch.mat'); % Load the test batch
testData = double(batch1.data(1:10000, :)) / 255.0; % Normalize test data
testLabels = batch1.labels(1:10000); % Extract corresponding test labels

% Convert the dataset to two classes 
class1 = 6; % First class for binary classification
class2 = 9; % Second class for binary classification

% Filter the training data for the selected classes
binaryTrainIdx = (trainLabels == class1) | (trainLabels == class2); % Find indices for the two classes
X_train = trainData(binaryTrainIdx, :); % Select corresponding training data
y_train = trainLabels(binaryTrainIdx); % Select corresponding training labels

% Relabel the binary training classes to 1 and -1
y_train(y_train == class1) = 1;
y_train(y_train == class2) = -1;

% Filter the test data for the selected classes
binaryTestIdx = (testLabels == class1) | (testLabels == class2); % Find indices for the two classes
X_test = testData(binaryTestIdx, :); % Select corresponding test data
y_test = testLabels(binaryTestIdx); % Select corresponding test labels

% Relabel the binary test classes to 1 and -1
y_test(y_test == class1) = 1;
y_test(y_test == class2) = -1;

% Compute PCA on the training data and retain 90% variance
[coeff, score, latent, tsquared, explained] = pca(X_train); 
cumulativeExplained = cumsum(explained); % Cumulative variance explained
numComponents = find(cumulativeExplained >= 90, 1); % Find number of components for 90% variance
X_train_pca = score(:, 1:numComponents); % Reduce dimensionality of training data

% Transform test data using the same PCA components
X_test_pca = (X_test - mean(X_train)) * coeff(:, 1:numComponents);

% Step 4: Train the SVM
SVMModel = fitcsvm(X_train_pca, y_train, 'KernelFunction', 'linear', 'BoxConstraint',0.01,'Standardize', true);
%{
hyperparameters = struct( ...
    'PolynomialOrder', [2, 3, 4], ... % Test different polynomial orders
    'BoxConstraint', [0.001, 0.01, 0.1, 1, 10]); % Test different C values

% Perform hyperparameter optimization for SVM with polynomial kernel
SVMModel = fitcsvm(X_train_pca, y_train, ...
    'KernelFunction', 'polynomial', ...
    'OptimizeHyperparameters', {'PolynomialOrder', 'BoxConstraint'}, ...
    'HyperparameterOptimizationOptions', struct( ...
        'AcquisitionFunctionName', 'expected-improvement-plus', ... % Bayesian optimization
        'MaxObjectiveEvaluations', 30)); % Number of evaluations to perform

% Display the best hyperparameters
disp(SVMModel.HyperparameterOptimizationResults.XAtMinObjective);cumulativeVariance
disp(SVMModel);
%}
%SVMModel = fitcsvm(X_train_pca, y_train,  'KernelFunction', 'polynomial', 'PolynomialOrder', 3, 'BoxConstraint', 0.1);
%SVMModel = fitcsvm(X_train_pca, y_train, 'KernelFunction', 'rbf','KernelScale' ,5,'BoxConstraint', 1,'Standardize',false);
%SVMModel = fitcsvm(X_train_pca, y_train,...
%'OptimizeHyperparameters', 'auto', ...
%    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', 'expected-improvement-plus'));
% 
% Predict on the test set
predictedLabels1 = predict(SVMModel, X_train_pca);
predictedLabels = predict(SVMModel, X_test_pca);
accuracy1 = mean(predictedLabels1 == y_train) * 100;
fprintf('Test Accuracy: %.2f%%\n', accuracy1);

% Calculate accuracy
accuracy = mean(predictedLabels == y_test) * 100;
fprintf('Test Accuracy: %.2f%%\n', accuracy);
