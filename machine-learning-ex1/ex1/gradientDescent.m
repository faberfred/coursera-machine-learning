function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
% GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% IMPORTANT: This is a BATCH gradient descent because ALL SAMPLES are involved in the process of improving the value of theta!!!
disp(size(X))
% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
    
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %h_x = X * theta; % this is a column-vector of size m x 1 
                      % => calculate the predictions for all x values -> these are the predicted y values
                      % this is done for all m samples at once
    
    %error = h_x - y; % this is a column-vector of size m x 1
                      % => calculate the error / distance between the prediction and the real value stored in y
                      % this is done for all m samples at once
    
    %b = error .* X;  % this is a matrix of size m x 2
                      % => multiply the error values element-wise with the values of X row by row
                      % match everey errot to the corresponding x values
    
    %c = sum(b);      % this is a row-vector of size 1 x 2 
                      % => the values of the rows within a column have been summed up
                      % This yields in a row-vector with 2 elements
    
    %delta = c / m;   % this is a row-vector of size 1 x 2 
                      % build the mean be dividing by m samples
                      % delta has to be transformed into a column-vector of size 2 x 1 
                      % because theta is also a column-vector of size 2 x 1 (theta = zeros(2, 1); % initialize fitting parameters)
    
    %theta = theta - (alpha * delta');  % this is a column-vector of size 2 x 1
    
    theta = theta - (alpha * ((sum(((X * theta) - y) .* X)) / m)');

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
