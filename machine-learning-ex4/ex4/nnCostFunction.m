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
X_plus_bias_one = [ones(size(X, 1), 1) X];

% calculate the activation values for the hidden layer  
Hidden_1 = sigmoid(X_plus_bias_one * Theta1'); 

% add the bias column to the hidden layer
Hidden_1 = [ones(size(Hidden_1, 1), 1) Hidden_1];

% calculate the activation values for the output layer
Output = sigmoid(Hidden_1 * Theta2'); % size of the output layer within this example = 5000 x 10 matrix

% init y_temp of size 5000 x 10
y_temp = zeros(size(Output)); % size of y_temp within this example = 5000 x 10 matrix

% transfer the label values of value 1 to 10 into logical arrays / vectors of site 1 x 10:
% value 1 will be transfered into the vector [1 0 0 0 0 0 0 0 0 0]
% value 2 will be transfered into the vector [0 1 0 0 0 0 0 0 0 0]
% value 3 will be transfered into the vector [0 0 1 0 0 0 0 0 0 0]
...
% value 10 will be transfered into the vector [0 0 0 0 0 0 0 0 0 1]
for i = 1:size(y, 1)
  y_temp(i, y(i)) = 1;
end;


% calculate the error (log loss function) for each of the (5000) input examples:
% the output is an array / a vectot of size 1 x 10. This vector will be compared with the label vector
% finally the resulting vector again of size 1 x 10 will be summed up to get one value for each of the 5000 examples. 
K_sum = sum((-y_temp .* log(Output) - (1 - y_temp) .* log(1 - Output)), 2);  
%K_sum = sum(-y_temp .* log(sigmoid(Hidden_1 * Theta2')) - (1 - y_temp) .* log(1 - sigmoid(Hidden_1* Theta2')));


%============ calculate the regularization term ============

% get rid of the bias column within Theta1 and Theta2
Theta1_origin = Theta1(:, 2:end);
Theta2_origin = Theta2(:, 2:end);

% sqare all elements within Theta1 and Theta2 (without the bias column)
Theta1_sq = Theta1_origin.^2;
Theta2_sq = Theta2_origin.^2;

% sum the first and second dimension of the theta matrices with the squared elements
Theta1_sum = sum(sum(Theta1_sq, 2));
Theta2_sum = sum(sum(Theta2_sq, 2));

% calculate the regulerazation value
reg_value = (lambda * (Theta1_sum + Theta2_sum)) / (2 * m);

% return the calculated cost value summed up over all the input elements (5000 elements in our example)
J = (sum(K_sum) / m) + reg_value;

%============== backpropagation part ===========================

Delta1 = zeros(hidden_layer_size, input_layer_size+1);
Delta2 = zeros(num_labels, hidden_layer_size+1);

% switch between voctorized and unvectorized version
vectorizedVersion = 1;

if(vectorizedVersion == 1)
  % ------ vectorized version -----------
  delta_3 = zeros(m, num_labels);
  delta_2 = zeros(m, hidden_layer_size);

  % calculate the activation values for the hidden layer
  z_2 = X_plus_bias_one * Theta1';
  Hidden_layer = sigmoid(z_2); 

  % add the bias column to the hidden layer
  Hidden_layer_bias = [ones(size(Hidden_layer, 1), 1) Hidden_layer];

  % calculate the activation values for the output layer
  z_3 = Hidden_layer_bias * Theta2';
  Output_layer = sigmoid(z_3);
  
  % calculete the delta value for the output layer
  delta_3 = Output_layer .- y_temp;
  % delta_3(i, :) = Output_layer - y_temp(i, :);

  % calculate the delta value for the hidden layer
  delta_2 = (delta_3 * Theta2_origin) .* sigmoidGradient(z_2);
  %delta_2(i, :) = (delta_3(i, :) * Theta2_origin) .* sigmoidGradient(z_2);

  % accumulate the gradients within the big delta matrix for all input values / examples (5000 in this case)
  % X_plus_bias_one(i, :) represents the input values of the i-th example -> this values are fix and will not be changed
  % Hidden_layer_bias represent the activation values of the hidden layer and will be computed for every input example.
  Delta1 = Delta1 .+ (delta_2' * X_plus_bias_one);
  Delta2 = Delta2 .+ (delta_3' * Hidden_layer_bias);
% -----------end of vectorized version ---------------------
else
% ---------- unvectorized version -------------
% calculate the gradient for each input example within a for loop
  for i = 1:m
    % calculate the activation values for the hidden layer
    z_2 = X_plus_bias_one(i, :) * Theta1';
    Hidden_layer = sigmoid(z_2); 
   
    % add the bias column to the hidden layer
    Hidden_layer_bias = [ones(size(Hidden_layer, 1), 1) Hidden_layer];
    
    % calculate the activation values for the output layer
    z_3 = Hidden_layer_bias * Theta2';
    Output_layer = sigmoid(z_3);
    
    % calculete the delta value for the output layer
    delta_3 = Output_layer - y_temp(i, :);
    % delta_3(i, :) = Output_layer - y_temp(i, :);
    
    % calculate the delta value for the hidden layer
    delta_2 = (delta_3 * Theta2_origin) .* sigmoidGradient(z_2);
    %delta_2(i, :) = (delta_3(i, :) * Theta2_origin) .* sigmoidGradient(z_2);
    
    % accumulate the gradients within the big delta matrix for all input values / examples (5000 in this case)
    % X_plus_bias_one(i, :) represents the input values of the i-th example -> this values are fix and will not be changed
    % Hidden_layer_bias represent the activation values of the hidden layer and will be computed for every input example.
    Delta1 = Delta1 .+ (delta_2' * X_plus_bias_one(i, :));
    Delta2 = Delta2 .+ (delta_3' * Hidden_layer_bias);
  end;
% ----------------- end of unvectorized version ---------------------
endif

% Obtain the (unregularized) gradient for the neural network cost function by dividing the accumulated gradients by m!
Theta1_grad = Delta1 / m;
Theta2_grad = Delta2 / m;

% Obtain the regularized gradient for the neural network cost function!
% ATTENTION: this update is just done for j >= 1. Therfore this will
% not be done for the first column of the matrix with the gradiants! 
% The first column is used for the bias term!
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda * Theta1(:, 2:end)) / m);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda * Theta2(:, 2:end)) / m);

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
