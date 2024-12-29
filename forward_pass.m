function [outputs, activations] = forward_pass(X, weights, biases)
    % X: input data
    % weights, biases: cell arrays containing weights and biases for each layer
   
    % Store activations and outputs for backpropagation
    activations = cell(length(weights), 1);
    outputs = cell(length(weights), 1);
    
    % Input layer
    inputs = X;
    for i = 1:length(weights)
        outputs{i} = inputs * weights{i} + biases{i};
        activations{i} = sigmoid(outputs{i});
        inputs = activations{i};  % Set input for next layer
    end
end
