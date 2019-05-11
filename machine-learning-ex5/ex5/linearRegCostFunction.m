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

% implementation of the unregularized cost function
%J = sum(((X * theta) - y).^2 ) / (2 * m);

% implementation of the regularized cost function
% exclude theta 0 from regularization!
theta_temp = theta(2:end, :);
J = (sum(((X * theta) - y).^2 ) / (2 * m)) + (lambda * sum(theta_temp.^2) / (2 * m));

% implementation of gradient without regularization
%grad = sum(((X * theta) - y) .* X) / m;

% implementation of gradient with regularization
% calculate the regularitation value
reg_temp = zeros(size(theta));
reg_temp1 = (lambda * theta) / m;
% exclude theta 0 from regularization!
reg_temp(2:end, :) = reg_temp1(2:end, :);

% calculate gradient with regularization
grad = (sum(((X * theta) - y) .* X) / m)' + reg_temp;

% =========================================================================

grad = grad(:);

end
