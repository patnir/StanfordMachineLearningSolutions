function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% The requirement for this was that no loops should be used to calculate
% the cost function or the gradient. Rather, a vectorized approach would be
% more appropriate. I have tried to break it down as much as possible.

J_without_reg = (-1* y' * log(sigmoid(X * theta))) + (-1* (1 - y)' * log(1 - sigmoid(X * theta)));

%Only utilizing indeces 2 through 28 for theta
J_reg = (lambda / (2)) * sum(theta(2:size(theta, 1), 1).^2);

J = (1 / m) * (J_without_reg + J_reg);

% Calculating gradient

grad_without_reg = (1 / m) .* ((sigmoid(X * theta) - y)' * X)';

% In order to not regularize theta(1), I decided to use a copy of theta and
% assign its first index's value to 0 so that it did not intefere with the
% calculations
holder = theta;
holder(1) = 0;

grad_reg = (lambda / m) * holder;

grad = grad_without_reg + grad_reg;








% =============================================================

grad = grad(:);

end
