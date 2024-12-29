function [outputs, activations] = forward_pass_2_hidden(X, weights, biases)
    % Forward pass through the first hidden layer
    outputs{1} = X * weights{1} + biases{1};         % Input to first hidden layer
    activations{1} = sigmoid(outputs{1});            % Activation (using sigmoid)

    % Forward pass through the second hidden layer
   outputs{2} = activations{1} * weights{2} + biases{2};  % Input to second hidden layer
    activations{2} = sigmoid(outputs{2});            % Activation (using sigmoid)

    % Forward pass through the output layer
    outputs{3} = activations{2} * weights{3} + biases{3};  % Input to output layer
    activations{3} = softmax(outputs{3});             % Activation (using softmax)
end
