
input_size = 32 * 32 * 3;   % 3072 features for CIFAR-10 images
hidden_size1 = 512;  % Number of neurons in the first hidden layer
hidden_size2 = 128; %Number of neurons in the second hidden layer
output_size = 10;            % Number of classes in CIFAR-10
lambda = 0.0001; % Regularization strength

% Initialize weights and biases with small random values
weights = {
   % randn(input_size, hidden_size1) * 0.01,   % Weights from input to hidden layer
   % randn(hidden_size1, hidden_size2) * 0.01,
   % randn(hidden_size2, output_size) * 0.01   % Weights from hidden to output layer
   randn(input_size, hidden_size1) * sqrt(2 / (input_size + hidden_size1))
   randn(hidden_size1, hidden_size2) * sqrt(2 / (hidden_size1 + hidden_size2))
   randn(hidden_size2, output_size) * sqrt(2 / (hidden_size2 + output_size))
};
biases = {
    zeros(1, hidden_size1),                 % Biases for hidden layer
    zeros(1, hidden_size2),       
    zeros(1, output_size)                    % Biases for output layer
};

learning_rate =0.01; % Initial learning rate, can be adjusted
num_epochs = 2600;        % Small number of epochs for testing
trainData = [];
trainLabels = [];
%{
% Load CIFAR-10 data
batch = load('data_batch_1.mat');   % Load only the first batch for simplicity
trainData = double(batch.data(1:8000, :)) / 255.0;  % Normalize pixel values to [0, 1]
trainLabels = batch.labels(1:8000);                 % First 1000 labels
% Compute the mean and standard deviation of the training data
%mean_train = mean(trainData, 1); % Compute mean for each feature
%std_train = std(trainData, 0, 1); % Compute standard deviation for each feature

% Normalize training data
%trainData = (trainData - mean_train) ./ std_train;
%}

num_Batches=5;
for i = 1:num_Batches
   batch = load(sprintf('data_batch_%d.mat', i)); % Load each batch
   trainData = [trainData; double(batch.data)/255.0];   % Concatenate data
   trainLabels = [trainLabels; batch.labels];     % Concatenate labels
end
%}
%mean_train = mean(trainData, 1); % Compute mean for each feature
%std_train = std(trainData, 0, 1); % Compute standard deviation for each feature

% Normalize training data
%trainData = (trainData - mean_train) ./ std_train;
% Convert labels to one-hot encoding for MLP
num_samples = size(trainData, 1);
trainLabelsOneHot = zeros(num_samples, output_size);
for i = 1:num_samples
   trainLabelsOneHot(i, trainLabels(i) + 1) = 1;  % +1 to match MATLAB 1-based indexing
end

% Load the test data (assuming test data is in 'test_batch.mat')
%testBatch = load('test_batch.mat');
%testData = double(testBatch.data) / 255.0;  % Normalize like training data
%testLabels = testBatch.labels;
%num_samples=1000;
batch1=load('test_batch.mat');
testData = double(batch1.data(1:10000, :)) / 255.0;  % Normalize pixel values to [0, 1]
testLabels = batch1.labels(1:10000);                 % First 1000 labels
batch_size = 32;  % Set the batch size (e.g., 64 samples per mini-batch)
num_batches = ceil(num_samples / batch_size);  % Calculate the number of mini-batches per epoch
%testData = (testData - mean_train) ./ std_train;

for epoch = 1:num_epochs
    for batch = 1:num_batches
        % Get the indices for the current mini-batch
        batch_start = (batch - 1) * batch_size + 1;
        batch_end = min(batch * batch_size, num_samples);
        X_batch = trainData(batch_start:batch_end, :);
        y_batch = trainLabelsOneHot(batch_start:batch_end, :);
        % Forward pass 
        [outputs, activations] = forward_pass_2_hidden(X_batch, weights, biases);
         
        % Backward pass 
        [d_weights, d_biases] = backward_pass(y_batch, activations, weights, outputs, X_batch);
        
        % Divide gradients by batch size
        for l = 1:length(d_weights)
            d_weights{l} = d_weights{l} / batch_size;
            d_biases{l} = d_biases{l} / batch_size;
        end
   %{
        % Update weights and biases with L2 regularization
       for l = 1:2
            weights{l} = weights{l} - learning_rate * (d_weights{l} + lambda * weights{l});
            biases{l} = biases{l} - learning_rate * d_biases{l};
       end
   %}
        % Update weights and biases with mini-batch gradients
       [weights, biases] = update_weights(weights, biases, d_weights, d_biases, learning_rate);

    end
    loss = cross_entropy_loss(y_batch, activations{end},weights);
    % Calculate and print training and test accuracy at the end of each epoch
    train_accuracy = calculate_accuracy(trainData, trainLabels, weights, biases);
    test_accuracy = calculate_accuracy(testData, testLabels, weights, biases);
     % Calculate loss     loss = cross_entropy_loss(y_batch, activations{end},weights);
    fprintf('Epoch %d,Loss: %.4f\n, Training Accuracy: %.2f%%, Test Accuracy: %.2f%%\n', epoch, loss,train_accuracy, test_accuracy);
end

%}
    








%Learning rate scheduler+losses diagrams
%{

% Initialize variables to store losses and accuracies
epoch_losses = zeros(1, num_epochs);
test_losses = zeros(1, num_epochs);
% Training loop
for epoch = 1:num_epochs
 if epoch>=160
           learning_rate=0.0005;
 elseif epoch>=350
           learning_rate=0.0001;

  end
    for batch = 1:num_batches
        
        % Get the indices for the current mini-batch
        batch_start = (batch - 1) * batch_size + 1;
        batch_end = min(batch * batch_size, num_samples);
        X_batch = trainData(batch_start:batch_end, :);
        y_batch = trainLabelsOneHot(batch_start:batch_end, :);

        % Forward pass with mini-batch data
        [outputs, activations] = forward_pass_2_hidden(X_batch, weights, biases);

        % Backward pass to compute gradients with mini-batch data
        [d_weights, d_biases] = backward_pass(y_batch, activations, weights, outputs, X_batch);
        
        % Divide gradients by batch size
        for l = 1:length(d_weights)
            d_weights{l} = d_weights{l} / batch_size;
            d_biases{l} = d_biases{l} / batch_size;
        end

        % Update weights and biases with mini-batch gradients
        [weights, biases] = update_weights(weights, biases, d_weights, d_biases, learning_rate);
    end

    % Calculate and print training and test accuracy at the end of each epoch
    train_accuracy = calculate_accuracy(trainData, trainLabels, weights, biases);
    test_accuracy = calculate_accuracy(testData, testLabels, weights, biases);
    
    % Calculate training and test loss
    [~,activations1]=forward_pass_2_hidden(trainData, weights, biases);
    [~,activations2]=forward_pass_2_hidden(testData, weights, biases);
  
testLabelsOneHot = zeros(size(testData, 1), output_size); % Ensure correct size
for i = 1:size(testData, 1)
    testLabelsOneHot(i, testLabels(i) + 1) = 1; % MATLAB index starts at 1
end


    train_loss = cross_entropy_loss(trainLabelsOneHot, activations1{end},weights);
    test_loss = cross_entropy_loss(testLabelsOneHot, activations2{end},weights);
    
    % Store losses for plotting
    epoch_losses(epoch) = train_loss;
    test_losses(epoch) = test_loss;

    % Print epoch results
    fprintf('Epoch %d, Loss: %.4f, Training Accuracy: %.2f%%, Test Accuracy: %.2f%%\n', ...
            epoch, train_loss, train_accuracy, test_accuracy);
end

% Plotting losses
figure;
plot(1:num_epochs, epoch_losses, '-o', 'LineWidth', 1.5, 'DisplayName', 'Training Loss');
hold on;
plot(1:num_epochs, test_losses, '-s', 'LineWidth', 1.5, 'DisplayName', 'Validation Loss');
xlabel('Epochs');
ylabel('Loss');
title('Training and Validation Loss over Epochs');
legend('Location', 'best');
grid on;
hold off;

%}











%DROP OUT

%{

% Initialize parameters
dropout_rate = 0.5;          % Dropout rate (e.g., 50% of neurons dropped)
dropout_active = true;       % Enable dropout during training
for epoch = 1:num_epochs
   
    for batch = 1:num_batches
        % Get the indices for the current mini-batch
        batch_start = (batch - 1) * batch_size + 1;
        batch_end = min(batch * batch_size, num_samples);
        X_batch = trainData(batch_start:batch_end, :);
        y_batch = trainLabelsOneHot(batch_start:batch_end, :);

        % Forward pass with dropout
        [outputs, activations] = forward_pass_2_hidden(X_batch, weights, biases);
        if dropout_active
            for l = 1:(length(activations) - 1)  % Apply dropout to hidden layers
    dropout_mask_1 = rand(size(activations{1})) > 0.2; % Input layer
    dropout_mask_2 = rand(size(activations{2})) > 0.5; % Hidden layer
    activations{1} = activations{1} .* dropout_mask_1 / 0.8;
    activations{2} = activations{2} .* dropout_mask_2 / 0.5;
           end
        end

        % Backward pass to compute gradients with mini-batch data
        [d_weights, d_biases] = backward_pass(y_batch, activations, weights, outputs, X_batch);

        % Divide gradients by batch size (mini-batch averaging)
        for l = 1:length(d_weights)
            d_weights{l} = d_weights{l} / batch_size;
            d_biases{l} = d_biases{l} / batch_size;
        end
  for l = 1:2
            weights{l} = weights{l} - learning_rate * (d_weights{l} + lambda * weights{l});
            biases{l} = biases{l} - learning_rate * d_biases{l};
       end
        % Update weights and biases
       % [weights, biases] = update_weights(weights, biases, d_weights, d_biases, learning_rate);
    end

    % Calculate training and validation metrics
    train_accuracy = calculate_accuracy(trainData, trainLabels, weights, biases);
    val_accuracy = calculate_accuracy(testData, testLabels, weights, biases);

    % Print metrics
    fprintf('Epoch %d, Training Accuracy: %.2f%%, Validation Accuracy: %.2f%%\n', epoch, train_accuracy, val_accuracy);
end

%}
%}
































% Calculate accuracy on training data
train_accuracy = calculate_accuracy(trainData, trainLabels, weights, biases);
fprintf('Training Accuracy: %.2f%%\n', train_accuracy);

% Load the test data (assuming test data is in 'test_batch.mat')
testBatch = load('test_batch.mat');
testData = double(testBatch.data) / 255.0;  % Normalize like training data
testLabels = testBatch.labels;

% Calculate accuracy on test data
test_accuracy = calculate_accuracy(testData, testLabels, weights, biases);
fprintf('Test Accuracy: %.2f%%\n', test_accuracy);























%%early stopping

% Initialize parameters
%max_epochs = 4000;          % Maximum limit for epochs
%patience = 30;              % Number of epochs to wait for test accuracy to improve
%best_test_accuracy = 0;     % Track the best test accuracy achieved
%no_improvement_count = 0;   % Counter for epochs with no improvement in test accuracy

%for epoch = 1:max_epochs
    % Perform forward pass and calculate training accuracy and loss
   % [train_outputs, train_activations] = forward_pass_2_hidden(trainData, weights, biases);
   % train_accuracy = calculate_accuracy(trainData, trainLabels, weights, biases);
    
    % Calculate test accuracy
   % test_accuracy = calculate_accuracy(testData, testLabels, weights, biases);
    
    % Print epoch results for monitoring
  %  fprintf('Epoch %d, Training Accuracy: %.2f%%, Test Accuracy: %.2f%%\n', ...
     %        epoch, train_accuracy, test_accuracy);

    % Check if test accuracy has improved
   % if test_accuracy > best_test_accuracy
     %   best_test_accuracy = test_accuracy;      % Update best test accuracy
       % no_improvement_count = 0;                % Reset no-improvement counter
    %else
       % no_improvement_count = no_improvement_count + 1;
   % end

    % Check stopping condition
   % if no_improvement_count >= patience && train_accuracy > best_test_accuracy
      %  fprintf('Stopping early at epoch %d due to lack of improvement in test accuracy.\n', epoch);
      %  break;
   % end

    % Backward pass and update weights
   % [d_weights, d_biases] = backward_pass(trainLabelsOneHot, train_activations, weights, train_outputs, trainData);
   % [weights, biases] = update_weights(weights, biases, d_weights, d_biases, learning_rate);
%end


