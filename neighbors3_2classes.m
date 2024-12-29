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

% Number of test samples
numTestSamples = size(testData, 1);

% Initialize counter for correct predictions
correct = 0;
predictedLabels = zeros(numTestSamples, 1);

% 3-Nearest Neighbors Classification
for j = 1:numTestSamples
    % Compute Euclidean distance from this test sample to all training samples
    distances = sum(abs(trainData - testData(j, :)), 2);%we don't need the sqrt because it will not make a difference in the min function

    % Find indices of the 3 smallest distances (3 nearest neighbors)
    [~, sortedIndices] = sort(distances);
    nearestNeighbors = sortedIndices(1:3);  

    % Retrieve the labels of the 3 nearest neighbors
    neighborLabels = trainLabels(nearestNeighbors);
    
    % Determine the most frequent label among the 3 neighbors 
     predictedLabel = mode(neighborLabels);
     predictedLabels(j) = predictedLabel;
    
    % Check if the predicted label matches the actual label
    % if predictedLabel == testLabels(j)
     %   correct = correct + 1;
    % end

     % Optional: Display progress
   if mod(j, 100) == 0
        fprintf('Processed %d / %d test samples\n', j, numTestSamples);
    end
end
% Calculate accuracy
accuracy = sum(predictedLabels == testLabels) / numTestSamples * 100;
fprintf('Accuracy of 3-Nearest Neighbor: %.2f%%\n', accuracy);

% Calculate accuracy
%accuracy = correct / numTestSamples * 100;
%fprintf('Accuracy of 3-Nearest Neighbors: %.2f%%\n', accuracy);