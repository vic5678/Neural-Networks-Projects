function svm_main()
    % Load CIFAR-10 dataset
    [trainData, trainLabels, testData, testLabels] = load_data();

    % Normalize data
    trainData = double(trainData) / 255;
    testData = double(testData) / 255;

    % Select and preprocess binary classification classes
    selectedClasses = [6, 9];
    trainIdx = ismember(trainLabels, selectedClasses);
    testIdx = ismember(testLabels, selectedClasses);

    trainData = trainData(trainIdx, :);
    trainLabels = preprocess_labels(trainLabels(trainIdx), selectedClasses);
    testData = testData(testIdx, :);
    testLabels = preprocess_labels(testLabels(testIdx), selectedClasses);

    % Train SVM
    regularization = 0.01;
    kernelType = 'linear';
    [weights, bias] = train_svm(trainData, trainLabels, kernelType, regularization);

    % Evaluate model
    trainError = compute_error(trainData, trainLabels, trainData, trainLabels, kernelType, weights, bias);
    testError = compute_error(trainData, trainLabels, testData, testLabels, kernelType, weights, bias);

    fprintf('Training Error: %.2f%%\n', trainError * 100);
    fprintf('Test Error: %.2f%%\n', testError * 100);
end

function [trainData, trainLabels, testData, testLabels] = load_data()
    trainBatch = load('data_batch_1.mat');
    testBatch = load('test_batch.mat');
    trainData = trainBatch.data;
    trainLabels = trainBatch.labels;
    testData = testBatch.data;
    testLabels = testBatch.labels;
end



