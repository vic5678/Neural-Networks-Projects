function predictions = predict_labels(trainData, trainLabels, testData, kernelType, weights, bias)
    scores = testData * weights + bias;
    predictions = sign(scores);
end
