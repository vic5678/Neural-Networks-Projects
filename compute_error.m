
function errorRate = compute_error(trainData, trainLabels, testData, testLabels, kernel, weights, bias)
    predictions = predict_labels(trainData, trainLabels, testData, kernel, weights, bias);
    errorRate = mean(predictions ~= testLabels);
end
