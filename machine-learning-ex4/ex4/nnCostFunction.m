function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% add the bias column to the input matrix X
X = [ones(size(X, 1), 1) X];
X_origin = X(:, 2:end);

% calculate the activation values for the hidden layer  
Hidden_1 = sigmoid(X * Theta1'); 

% add the bias column to the hidden layer 1
Hidden_1 = [ones(size(Hidden_1, 1), 1) Hidden_1];

% calculate the activation values for the output layer
Output = sigmoid(Hidden_1 * Theta2'); % size of the output layer within this example = 5000 x 10 matrix

y_temp = zeros(size(Output)); % size of y_temp within this example = 5000 x 10 matrix

for i = 1:size(y, 1)
  y_temp(i, y(i)) = 1;
end;
%disp(y_temp(1:10, :));

% calculate the error for each impot example.
K_sum = sum((-y_temp .* log(Output) - (1 - y_temp) .* log(1 - Output)), 2);  
%K_sum = sum(-y_temp .* log(sigmoid(Hidden_1 * Theta2')) - (1 - y_temp) .* log(1 - sigmoid(Hidden_1* Theta2')));


% get rid of the bis column within Theta1 and Theta2
Theta1_origin = Theta1(:, 2:end);
Theta2_origin = Theta2(:, 2:end);

% sqare all elements within Theta1 and Theta2
Theta1_sq = Theta1_origin.^2;
Theta2_sq = Theta2_origin.^2;

% sum the first and second dimension of the squared theta matrices 
Theta1_sum = sum(sum(Theta1_sq));
Theta2_sum = sum(sum(Theta2_sq));

% calculate the regulerazation value
reg_value = (lambda * (Theta1_sum + Theta2_sum)) / (2 * m);

% return the calculated cost value
J = (sum(K_sum) / m) + reg_value;

% -------------------------------------------------------------
% backpropagation part:

%delta_3 = zeros(m, num_labels);
%delta_2 = zeros(m, hidden_layer_size);

Delta1 = zeros(hidden_layer_size, input_layer_size+1);
Delta2 = zeros(num_labels, hidden_layer_size+1);

%disp(size(delta_out));

for i = 1:m
  % calculate the activation values for the hidden layer
  z_2 = X(i, :) * Theta1';
  Hidden_layer = sigmoid(z_2); 
  %disp(size(Hidden_layer)); 
 
  % add the bias column to the hidden layer
  Hidden_layer_bias = [ones(size(Hidden_layer, 1), 1) Hidden_layer];
  %disp(size(Hidden_layer)); 
  
  % calculate the activation values for the output layer
  z_3 = Hidden_layer_bias * Theta2';
  Output_layer = sigmoid(z_3);
  %disp(Output_layer);
  %disp(y_temp(i, :));
  
  delta_3 = Output_layer - y_temp(i, :);
  % delta_3(i, :) = Output_layer - y_temp(i, :);
  %disp(delta_3(i, :));
  
  delta_2 = (delta_3 * Theta2_origin) .* sigmoidGradient(z_2);
  %delta_2(i, :) = (delta_3(i, :) * Theta2_origin) .* sigmoidGradient(z_2);
  %disp(size(X_origin(i, :)));
  %disp(size(delta_2' * X_origin(i, :)));
  
  %Delta1 = Delta1 .+ (delta_2' * X_origin(i, :));
  %Delta2 = Delta2 .+ (delta_3' * Hidden_layer);
  
  Delta1 = Delta1 .+ (delta_2' * X(i, :));
  Delta2 = Delta2 .+ (delta_3' * Hidden_layer_bias);
  
end;

Theta1_grad = Delta1 / m;
Theta2_grad = Delta2 / m;

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda * Theta1(:, 2:end)) / m);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda * Theta2(:, 2:end)) / m);

%disp(size(Theta1_grad));

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end