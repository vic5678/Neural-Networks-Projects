function value = compute_kernel(x1, x2, kernelType)
    switch kernelType
        case 'linear'
            value = x1 * x2';
        case 'poly'
            degree = 3; % Polynomial degree
            value = (x1 * x2' + 1)^degree;
        case 'rbf'
            sigma = 2; % Gaussian width
            value = exp(-norm(x1 - x2)^2 / (2 * sigma^2));
        otherwise
            error('Unsupported kernel.');
    end
end