function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% finding all negative and positive values
positive_values = find(y == 1);
negative_values = find(y == 0);

hold on

% need to plot which value corresponds with which student passing or
% failing

plot(X(positive_values, 1), X(positive_values, 2), 'r+');
plot(X(negative_values, 1), X(negative_values, 2), 'bo');

hold off






% =========================================================================



hold off;

end
