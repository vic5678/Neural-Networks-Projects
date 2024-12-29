function out = sigmoid_derivative(x)
         out =  sigmoid(x) .* (1 - sigmoid(x));
end