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

% Define sigmoid and its derivative
sigmoid = @(x) 1 ./ (1 + exp(-x));
sigmoid_derivative = @(x) sigmoid(x) .* (1 - sigmoid(x));

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
        
        % Forward pass (train)
        hidden_layer1 = X_batch * weights{1} + biases{1};
        activations{1} = sigmoid(hidden_layer1); % Sigmoid activation

        output_layer = activations{1} * weights{2} + biases{2};
        activations{2} = output_layer; % Linear output for scores

        % Compute hinge loss
        margins = max(0, 1 - y_batch .* activations{2});
        loss = sum(margins) / size(X_batch, 1);

        % Backward pass
        delta = -(y_batch .* (margins > 0)); % Error term for output layer
        d_weights = cell(2, 1);
        d_biases = cell(2, 1);

        % Gradients for output layer
        d_weights{2} = activations{1}' * delta / size(X_batch, 1);
        d_biases{2} = sum(delta, 1) / size(X_batch, 1);

        % Backpropagate to hidden layer
        delta = (delta * weights{2}') .* sigmoid_derivative(hidden_layer1);
        d_weights{1} = X_batch' * delta / size(X_batch, 1);
        d_biases{1} = sum(delta, 1) / size(X_batch, 1);

        % Update weights and biases
        for l = 1:length(weights)
            weights{l} = weights{l} - learning_rate * d_weights{l};
            biases{l} = biases{l} - learning_rate * d_biases{l};
        end
    end

    % Compute train accuracy
    hidden_layer1 = trainData * weights{1} + biases{1};
    activations_train = sigmoid(hidden_layer1);
    output_train = activations_train * weights{2} + biases{2};
    train_predictions = output_train > 0;
    train_accuracy = mean(train_predictions == (binaryLabelsTrain > 0));

    % Forward pass (test)
    hidden_layer1_test = testData * weights{1} + biases{1};
    activations_test = sigmoid(hidden_layer1_test);
    output_test = activations_test * weights{2} + biases{2};
    test_predictions = output_test > 0;
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


