function [weights, biases] = update_weights(weights, biases, d_weights, d_biases, eta)
    for l = 1:length(weights)
        weights{l} = weights{l} - eta* d_weights{l};
        biases{l} = biases{l} - eta* d_biases{l};
    end
end
