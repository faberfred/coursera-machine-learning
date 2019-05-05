function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% add the bias column to the input matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% calculate the values for the knodes of the hidden layer
% use the sigmoid function as the activation function
hid = sigmoid(X * Theta1');

% get the size / amount of the rows of the hidden matrix 
hid_size = size(hid, 1);

% add the bias column to the hidden matrix
hid = [ones(hid_size, 1) hid];

% calculate the values for the knodes of the outpot layer
% use the sigmoid function as the activation function
out = sigmoid(hid * Theta2');

% get the maximum value and its index for each row 
% the index of the max value indicates the predicted digit (0 (=10) to 9)
[val, ix] = max(out, [], 2);

% assign the index values (=predicted digits) to the return value
p = ix;

% =========================================================================


end
