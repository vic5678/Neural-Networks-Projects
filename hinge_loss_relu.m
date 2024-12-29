% Define parameters
input_size = 32 * 32 * 3;  % 3072 features for CIFAR-10 images
hidden_size1 = 128;        % Number of neurons in the hidden layer
output_size = 1;           % Binary classification (1 output neuron)
learning_rate = 0.01;      % Learning rate
num_epochs = 2600;         % Number of epochs
batch_size = 128;          % Batch size for mini-batch gradient descent

% Initialize weights and biases
weights = {
    randn(input_size, hidden_size1) * sqrt(2 / (input_size + hidden_size1)),
    randn(hidden_size1, output_size) * sqrt(2 / (hidden_size1 + output_size))
};
biases = {
    zeros(1, hidden_size1),
    zeros(1, output_size)
};

% Load and preprocess training data
num_batches = 5;
trainData = [];
trainLabels = [];
for i = 1:num_batches
    batch = load(sprintf('data_batch_%d.mat', i));
    trainData = [trainData; double(batch.data) / 255.0]; % Normalize pixel values
    trainLabels = [trainLabels; batch.labels];
end

% Load and preprocess test data
test_batch = load('test_batch.mat');
testData = double(test_batch.data) / 255.0;
testLabels = test_batch.labels;

% Filter two classes for binary classification
class1 = 6; % Example class 1
class2 = 9; % Example class 2
binaryIdxTrain = (trainLabels == class1) | (trainLabels == class2);
binaryIdxTest = (testLabels == class1) | (testLabels == class2);

trainData = trainData(binaryIdxTrain, :);
trainLabels = trainLabels(binaryIdxTrain);
testData = testData(binaryIdxTest, :);
testLabels = testLabels(binaryIdxTest);

binaryLabelsTrain = double(trainLabels == class1);
binaryLabelsTrain(binaryLabelsTrain == 0) = -1; % Convert to {-1, 1}

binaryLabelsTest = double(testLabels == class1);
binaryLabelsTest(binaryLabelsTest == 0) = -1; % Convert to {-1, 1}

% Training loop
num_samples = size(trainData, 1);
num_batches_per_epoch = ceil(num_samples / batch_size);

for epoch = 1:num_epochs
    for batch_idx = 1:num_batches_per_epoch
        % Get batch data
        start_idx = (batch_idx - 1) * batch_size + 1;
        end_idx = min(batch_idx * batch_size, num_samples);
        X_batch = trainData(start_idx:end_idx, :);
        y_batch = binaryLabelsTrain(start_idx:end_idx);

        % Forward pass
        hidden_layer1 = max(0, X_batch * weights{1} + biases{1}); % ReLU activation
        scores_train = hidden_layer1 * weights{2} + biases{2};

        % Compute hinge loss
        margins = max(0, 1 - y_batch .* scores_train);
        loss = sum(margins) / size(X_batch, 1);

        % Backward pass
        dscores = -(y_batch .* (margins > 0));
        grads_w2 = hidden_layer1' * dscores / size(X_batch, 1);
        grads_b2 = sum(dscores, 1) / size(X_batch, 1);

        dhidden1 = dscores * weights{2}' .* (hidden_layer1 > 0); % ReLU derivative
        grads_w1 = X_batch' * dhidden1 / size(X_batch, 1);
        grads_b1 = sum(dhidden1, 1) / size(X_batch, 1);

        % Update weights and biases
        weights{2} = weights{2} - learning_rate * grads_w2;
        biases{2} = biases{2} - learning_rate * grads_b2;

        weights{1} = weights{1} - learning_rate * grads_w1;
        biases{1} = biases{1} - learning_rate * grads_b1;
    end

    % Compute train accuracy
    hidden_layer1 = max(0, trainData * weights{1} + biases{1});
    scores_train = hidden_layer1 * weights{2} + biases{2};
    train_predictions = scores_train > 0;
    train_accuracy = mean(train_predictions == (binaryLabelsTrain > 0));

    % Forward pass (test)
    hidden_layer1_test = max(0, testData * weights{1} + biases{1});
    scores_test = hidden_layer1_test * weights{2} + biases{2};
    test_predictions = scores_test > 0;
    test_accuracy = mean(test_predictions == (binaryLabelsTest > 0));

    % Print loss and accuracies every 100 epochs
    if mod(epoch, 100) == 0
        fprintf('Epoch %d: Loss = %.4f\n', epoch, loss);
        fprintf('Epoch %d: Training Accuracy = %.2f%%, Test Accuracy = %.2f%%\n', epoch, train_accuracy * 100, test_accuracy * 100);
    end
end

% Final accuracy
fprintf('Final Training Accuracy: %.2f%%\n', train_accuracy * 100);
fprintf('Final Test Accuracy: %.2f%%\n', test_accuracy * 100);
