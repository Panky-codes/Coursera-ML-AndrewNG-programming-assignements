function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
z_size = size(z);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

for i = 1:z_size(1)
	for j = 1:z_size(2)
		g(i,j)= 1/(1+e^(-z(i,j)));
	endfor
endfor

%===========================================================
%Alternative matrix implementation
%g = 1./(1+e^(-z))


% =============================================================

end
