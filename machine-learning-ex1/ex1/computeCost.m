function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% theta is a [n x 1] dimensional matrix, where n is the number of
% parameters. Therefore, n rows and 1 column
% 
% X is a [m x n] dimensional matrix, where m is the number of training
% examples. Therefore dimensions will agree when you multiply
% [m x n] * [n x 1]
predictions = X * theta;

errors_squared = (predictions - y) .^ 2;

J = (1 / (2 * m)) * sum(errors_squared);


% =========================================================================

end
