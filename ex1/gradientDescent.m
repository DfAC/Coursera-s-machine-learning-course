function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

  % cost J(?) is parameterized by the vector ?, not X and y
  v = X*theta-y;
  delta = (v'*X)/m;
  %update value with prediction
   theta = theta -alpha.*delta'; %update all values in the same time
   % Save the cost J in every iteration    
   J_history(iter) = computeCost(X, y, theta);

end
a=1;

end
