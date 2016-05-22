function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% The cost function with be divided into two parts
h_x = X * theta;

J_without_reg = (1 / (2 * m)) * sum((h_x - y) .^ 2);
J_with_reg = (lambda / (2 * m)) * sum(theta(2:end, :) .^2);

J = J_without_reg + J_with_reg;

% Finding the gradient

theta_gradient = theta;
theta_gradient(1, :) = zeros(1, size(theta_gradient, 2));

grad = ((1 / m) * ((h_x - y)' * X) + (lambda / m) * theta_gradient');
% =========================================================================

grad = grad(:);

end
