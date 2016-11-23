function g = sigmoidGradient(z)
g = zeros(size(z));

sigmoid = sigmoid(z);
g = sigmoid .* (1 .- sigmoid);

end
