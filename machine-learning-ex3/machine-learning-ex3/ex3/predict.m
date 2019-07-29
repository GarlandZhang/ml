function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

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


biased = [ones(m, 1) X];
for ex = 1:m
  units_1 = zeros(rows(Theta1), 1);
  % second layer
  input_x = biased(ex,:);
  for param_i = 1:rows(Theta1)
    theta1_i = Theta1(param_i, :)';
    z = input_x * theta1_i;
    a = sigmoid(z);
    units_1(param_i) = a;
  end

  % third layer
  units_1 = [1 units_1']; % add bias and make row vector

  units_2 = zeros(rows(Theta2), 1);

  for param_i = 1:rows(Theta2)
    theta2_i = Theta2(param_i, :)';
    z = units_1 * theta2_i;
    a = sigmoid(z);
    units_2(param_i) = a;
  end

  [m, prediction] = max(units_2); % scalar value
  p(ex) = prediction;

end





% =========================================================================


end
