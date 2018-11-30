function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
theta_size = size(theta);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%===================================================================
%For loop implementation - less efficient
%Reg_value = (lambda/(2*m))*(sum(theta(2:theta_size(1)).*theta(2:theta_size(1))));



%for i = 1:m
%J = J + (-y(i)*log(sigmoid(X(i,:)*theta)) - ((1-y(i))*log(1-sigmoid(X(i,:)*theta))));
%endfor
%J = (J/m) + Reg_value;

%Calculation of Grad


%for j = 1:theta_size(1)
%	temp_grad = 0;
%	for k = 1:m
%		temp_grad = temp_grad + (sigmoid(X(k,:)*theta) - y(k)) * X(k,j);
%	endfor 
%	if (j==1)
%		grad(j,1) = temp_grad/m;
%	else
%		grad(j,1) = temp_grad/m + ((lambda/m)*theta(j,1));
%	endif
%endfor


% ===========================================================
%Vectorized Implementation - More efficient
J = (1/m)* (-y' * log(sigmoid(X*theta)) - (1-y)' * log(1-sigmoid(X*theta)));

grad =  ((1/m) * X' * (sigmoid(X*theta) - y)) + (lambda/m) * theta   ;
grad(1) = grad(1) - (lambda/m) * theta(1) ;

Reg_value = (lambda/(2*m))*(sum(theta(2:theta_size(1)).*theta(2:theta_size(1))));


J = J + Reg_value;


% =============================================================

end
