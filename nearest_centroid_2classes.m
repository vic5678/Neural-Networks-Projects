 %Define the two classes for binary classification
class1 = 3; % Change to desired class 1
class2 = 5; % Change to desired class 2

% Load CIFAR-10 training batches
numBatches = 5; % CIFAR-10 has 5 training batches
trainData = [];
trainLabels = [];

for i = 1:numBatches
    batch = load(sprintf('data_batch_%d.mat', i)); % Load each batch
    trainData = [trainData; double(batch.data)];   % Concatenate data
    trainLabels = [trainLabels; batch.labels];     % Concatenate labels
end

% Filter for the two selected classes
binaryTrainIdx = (trainLabels == class1) | (trainLabels == class2);
trainData = trainData(binaryTrainIdx, :);
trainLabels = trainLabels(binaryTrainIdx);

% Convert labels to binary (-1, +1)
trainLabels(trainLabels == class1) = -1;
trainLabels(trainLabels == class2) = +1;

% Print dataset info
fprintf('Number of binary training samples: %d\n', size(trainData, 1));

% Load CIFAR-10 test batch
testBatch = load('test_batch.mat');
testData = double(testBatch.data);
testLabels = testBatch.labels;

% Filter test data for the two selected classes
binaryTestIdx = (testLabels == class1) | (testLabels == class2);
testData = testData(binaryTestIdx, :);
testLabels = testLabels(binaryTestIdx);

% Convert test labels to binary (-1, +1)
testLabels(testLabels == class1) = -1;
testLabels(testLabels == class2) = +1;

% Print dataset info
numTestSamples = size(testData, 1);
fprintf('Number of binary test samples: %d\n', numTestSamples);

% Find unique class labels
classLabels = unique(trainLabels);

% Calculate centroids for each class
numClasses = length(classLabels);
centroids = zeros(numClasses, size(trainData, 2)); % Each row will store the centroid of a class

for k = 1:numClasses
    % Get all training samples belonging to the k-th class
    classData = trainData(trainLabels == classLabels(k), :);
    
    % Compute the centroid (mean) of the k-th class
    centroids(k, :) = mean(classData, 1);
end

% Nearest Class Centroid Classification
correct = 0; % Counter for correct predictions
numTestSamples = size(testData, 1);

for j = 1:numTestSamples
    % Compute Euclidean distance from the test sample to each class centroid
    distances = sum((centroids - testData(j, :)).^2, 2);
    
    % Find the index of the nearest centroid
    [~, minIndex] = min(distances);
    
    % Predict the label of the nearest centroid
    predictedLabel = classLabels(minIndex);
    
    % Check if the predicted label matches the actual label
    if predictedLabel == testLabels(j)
        correct = correct + 1;
    end

end

% Calculate accuracy
accuracy = correct / numTestSamples * 100;
fprintf('Accuracy of Nearest Class Centroid: %.2f%%\n', accuracy);