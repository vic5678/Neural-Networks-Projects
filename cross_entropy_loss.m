function loss = cross_entropy_loss(y_true, y_pred,weights)
    % y_true should be one-hot encoded
    % y_pred should be the softmax output for multi-class classification
    y_pred = max(min(y_pred, 1 - 1e-15), 1e-15);  % Avoid log(0)
    loss = -mean(sum(y_true .* log(y_pred), 2));
    lambda=0.0001;
%{
% Compute the L2 regularization term

l2_penalty = 0;
for l = 1:2
    l2_penalty = l2_penalty + sum(weights{l}(:).^2);
end
l2_penalty = (lambda / 2) * l2_penalty;

% Total loss with L2 regularization
total_loss = loss + l2_penalty;
%}
end