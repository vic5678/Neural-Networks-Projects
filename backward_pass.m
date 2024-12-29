function [d_weights, d_biases] = backward_pass(y_true, activations, weights, outputs,X)
    num_layers = length(weights);
    d_weights = cell(num_layers, 1);
    d_biases = cell(num_layers, 1);
    
    % Error term for the output layer
    delta = (activations{end} - y_true);
    
    % Backpropagation through layers
    for l = num_layers:-1:1
        if l > 1
            d_weights{l} =  activations{l-1}' *delta;  % Gradient for weights
        else
            d_weights{l} = X' * delta;
        end
        d_biases{l} = sum(delta, 1);                  % Gradient for biases
        if l > 1
            delta = (delta * weights{l}') .* sigmoid_derivative(outputs{l-1});
        end
    end
end
