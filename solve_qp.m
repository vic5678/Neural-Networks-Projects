function alpha = solve_qp(H, labels, C)
    numSamples = length(labels);
    options = optimoptions('quadprog', 'Display', 'none');
    % Linear term
    c = -ones(numSamples, 1);
    % Add small regularization to avoid numerical issues
    H = H + 1e-10 * eye(numSamples);
    lb = zeros(numSamples, 1); % Lower bound for alpha
    ub = C * ones(numSamples, 1); % Upper bound for alpha
    x0 = zeros(numSamples, 1); % Initial guess for alpha
    alpha = quadprog(H, c, [], [], labels', 0, lb, ub, x0, options);
end
