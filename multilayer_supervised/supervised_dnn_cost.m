function [cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
gradCache = cell(numHidden+1);
%% forward prop
%%% YOUR CODE HERE %%%
[m, N] = size(data); 
C = ei.output_dim;

hAct{1} = data;
regPenalty = 0;
output_layer = zeros(C, N);

for i=1:numHidden+1
  W = stack{i}.W;
  b = stack{i}.b;
  a = hAct{i};
  z = W * a + b;            %% broadcasting

  gradCache{i} = [];
  gradCache{i}.dzda = W';
  gradCache{i}.dzdW = a';
  gradCache{i}.dzdb = 1;

  if(i ~= numHidden + 1)
    switch(ei.activation_fun)
      case('logistic')
        f = sigmoid(z);
        gradCache{i}.dfdz = f .* (1 - f);
      case('tanh')
        f = 2 * sigmoid(2 * z) - 1;
        gradCache{i}.dfdz = 1 - f.^2;
      case('relu')
        f = max(0, z);
        gradCache{i}.dfdz = (f > 0);
      otherwise
        fprintf('ERROR: Activatio Function %s not supported', ei.activation_fun);
        return;
     endswitch
    hAct{i+1} = f;

  else
    output_layer = z;
  end;
  
  regPenalty = regPenalty + sum(sum(W.^2));
end;

[sigma, dSoftdScores, pred_prob] = soft_max(output_layer, labels);

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
ceCost = -sum(sigma);
wCost = ei.lambda / 2 * regPenalty;
pCost = 0;

cost = ceCost + wCost + pCost;
dLdout = -dSoftdScores;

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
accumulated_grad = dLdout;
for i = numHidden+1:-1:1
  W = stack{i}.W;
  cache = gradCache{i};
  gradStack{i} = [];
  if(i ~= numHidden + 1)
    dLdz = (accumulated_grad .* cache.dfdz);
    gradStack{i}.W = dLdz * cache.dzdW + ei.lambda * W;
    gradStack{i}.b = sum(dLdz, 2);
    accumulated_grad = cache.dzda * dLdz;
  else
    gradStack{i}.W = accumulated_grad * cache.dzdW + ei.lambda * W;
    gradStack{i}.b = sum(accumulated_grad * cache.dzdb, 2);
    accumulated_grad = cache.dzda * accumulated_grad;
  end
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end

function [sigma, dSoftdScores, prob] = soft_max(class_scores, class_labels)
sigma = [];
dSoftdScores = [];
K = max(class_scores);

num = exp(class_scores - K);  % numerical stability
denom = sum(num, 1);
prob = num ./ denom;

if(~isempty(class_labels))
  indices = 1:size(class_labels, 1);
  y = sub2ind(size(num), class_labels, indices(:));
  sigma = log(prob(y));

  one_hot_mat = zeros(size(num));
  one_hot_mat(y) = 1;
  dSoftdScores = (one_hot_mat - prob);
end;
end

