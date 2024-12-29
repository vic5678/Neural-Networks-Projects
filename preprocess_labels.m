function labels = preprocess_labels(labels, targetClasses)
    labels = double(labels == targetClasses(1)) * 2 - 1;
end