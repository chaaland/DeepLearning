function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 
[num_features, num_examples] = size(data);

% Forward pass (compute activations)
z2 = W1 * data + b1;      % Broadcasting
[a2, da2dz2] = sigmoid(z2);
z3 = W2 * a2 + b2;        % Broadcasting
[a3, da3dz3] = sigmoid(z3);

% Compute Costs
L1 = 0.5 * sum(sum((data - a3).^2));
L2 = lambda / 2 * (sum(sum(W1.^2)) + sum(sum(W2.^2)));
rho_hat = 1/num_examples * sum(a2, 2);
[L3, dKLdrho_hat] = bern_KL_div(sparsityParam, rho_hat, beta);
cost += L1 + L2 + L3;

dL1da3 = -(data - a3);
dL1dz3 = dL1da3 .* da3dz3;

dL1dW2 = dL1dz3 * a2.';
dL1db2 = sum(dL1dz3, 2);

dL1da2 = W2.' * dL1dz3;
dL1dz2 = dL1da2 .* da2dz2;

dL1dW1 = dL1dz2 * data.';
dL1db1 = sum(dL1dz2, 2);

dL2dW2 = lambda * W2;
dL2dW1 = lambda * W1;

dL3da2 = 1/num_examples * dKLdrho_hat; 
dL3dz2 = dL3da2 .* da2dz2;

dL3dW1 = dL3dz2 * data.';
dL3db1 = sum(dL3dz2, 2);

W2grad = dL1dW2 + dL2dW2;
b2grad = dL1db2;

W1grad =  dL1dW1 + dL2dW1 + dL3dW1;
b1grad =  dL1db1 + dL3db1;


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function [h, dhdx] = sigmoid(x)
  
    h = 1 ./ (1 + exp(-x));
    dhdx = h .* (1 - h);
end

function [kl, dKLdrho_hat] = bern_KL_div(rho, rho_hat, beta);
  kl = beta * sum(rho * log(rho ./ rho_hat) + (1 - rho) * log((1 - rho) ./ (1 - rho_hat)));
  dKLdrho_hat = beta * (-rho ./ rho_hat + (1 - rho) ./ (1 - rho_hat));
end