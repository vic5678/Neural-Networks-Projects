function accuracy = calculate_accuracy(data, labels, weights, biases)
    % Perform a forward pass to get the network output fot the current data
    [~, activations] = forward_pass(data, weights, biases);
    predictions = activations{end};  % Get the output layer activations
    
    % For each sample, select the index of the max probability as the predicted class
    [~, predicted_classes] = max(predictions, [], 2);
    
    % Convert true labels to match the MATLAB format of predicted classes
    %( 1 for class 0, 2 for class 1 in MATLAB etc)
    true_classes = labels +1;  
    
    % Calculate accuracy as the percentage of correct predictions
    accuracy = mean(predicted_classes == true_classes) * 100;
end
