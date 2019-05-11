function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
x1 = X(:, 1)'; x2 = X(:, 2)';
C_test = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_test = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
prediction_error_map = ones(3, 1);

for i = 1:length(C_test)
  for j = 1:length(sigma_test)
    %disp(C_test(i));
    %disp(sigma_test(j));
    model= svmTrain(X, y, C_test(i), @(x1, x2) gaussianKernel(x1, x2, sigma_test(j)));
    pred = svmPredict(model, Xval);
    prediction_error = mean(double(pred ~= yval));
    %disp(prediction_error);
    if (prediction_error <= prediction_error_map(1))
      prediction_error_map(1) = prediction_error;
      prediction_error_map(2) = C_test(i);
      prediction_error_map(3) = sigma_test(j);
    endif;
  end;
end;

C = prediction_error_map(2);
sigma = prediction_error_map(3);
%disp(C);
%disp(sigma);
% =========================================================================

end
