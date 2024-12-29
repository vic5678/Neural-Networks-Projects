function y = softmax(x)
    % Subtract the maximum value from each row for numerical stability
    % This avoids large exponentials that could lead to overflow
    x = x - max(x, [], 2);

    exp_x = exp(x);

    % Normalize by dividing each element by the sum of exponentials in its row
    y = exp_x ./ sum(exp_x, 2);
end
