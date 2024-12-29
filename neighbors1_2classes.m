% Define the two classes for binary classification
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

% Initialize predictions
predictedLabels = zeros(numTestSamples, 1);

% 1-Nearest Neighbor Classification
for i = 1:numTestSamples
    % Compute Euclidean distance to all training samples
    distances = sum((trainData - testData(i, :)).^2, 2); % Avoid sqrt for efficiency
    
    % Find the index of the minimum distance
    [~, minIndex] = min(distances);
    
    % Assign label of the nearest neighbor
    predictedLabels(i) = trainLabels(minIndex);
    
    % Optional: Display progress
    if mod(i, 100) == 0
        fprintf('Processed %d / %d test samples\n', i, numTestSamples);
    end
end

% Calculate accuracy
accuracy = sum(predictedLabels == testLabels) / numTestSamples * 100;
fprintf('Accuracy of 1-Nearest Neighbor (Binary Classification): %.2f%%\n', accuracy);
