function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
X= [ones(size(X,1), 1) X];
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
Z_two = Theta1 * X';
a_two = sigmoid(Z_two);
a_two = a_two';
a_two = [ones(size(a_two,1), 1) a_two];
a_two = a_two';
Z_three = Theta2 * a_two;
h_three = sigmoid(Z_three);
h_three = h_three';
[max_values p] = max(h_three,[],2);







% =========================================================================


end
