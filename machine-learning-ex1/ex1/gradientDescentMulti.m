function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    theta_t = theta';
    temp_theta = zeros(size(theta, 1), 1);
    
    for i = 1 : size(theta, 1)
      
      temp_sum = zeros(size(theta, 1), 1);
      
      for j = 1 : m
        
        temp_x = X(j,:)';
        temp_sum(i, 1) = temp_sum(i, 1) + (((theta_t * temp_x) - y(j)) * X(j , i));
        
      endfor
      
      temp_sum(i, 1) = (temp_sum(i, 1) * alpha) / m;
      temp_theta(i, 1) = theta(i) - temp_sum(i, 1);
      
    endfor

    theta = temp_theta;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
