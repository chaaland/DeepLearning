%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);
[m, n] = size(x);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

epsilon = 1e-2;
z = W * x;
y = W.' * z;
w = (y - x);

a = sqrt(z.^2 + epsilon);
b = sum(a);
c = sum(b);

L1 = params.lambda * c;

L2 = 0.5 * sum(sum(w.^2));
cost = L1 + L2;

gradL1 = params.lambda * (z ./ a) * x.';
gradL2 = (z * w.' + (W * w) * x.');
Wgrad = gradL1 + gradL2;

% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);