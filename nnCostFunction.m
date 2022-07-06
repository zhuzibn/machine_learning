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

X=[ones(m,1) X];
z2=Theta1*X';
a2=sigmoid(z2);
a2=[ones(1,m);a2];
z3=Theta2*a2;
a3=sigmoid(z3);

y_tmp=zeros(num_labels,m);
for ct2=1:m
    if y(ct2)==1
        y_tmp(:,ct2)=[1;zeros(num_labels-1,1)];
    elseif y(ct2)==num_labels
        y_tmp(:,ct2)=[zeros(num_labels-1,1);1];
    else
        y_tmp(:,ct2)=[zeros(y(ct2)-1,1);1;zeros(num_labels-y(ct2),1)];
    end
end
if lambda
    J=sum(sum(-y_tmp.*log(a3)-(1-y_tmp).*log(1-a3)))/m+lambda/(2*m)*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));
else
    
    J=sum(sum(-y_tmp.*log(a3)-(1-y_tmp).*log(1-a3)))/m;
end
    DELT2=0;
    DELT1=0;
for ct1=1:m
    a1=X(ct1,:);  
    %a1=[1 a1];
    z2=Theta1*a1';
    a2=sigmoid(z2);
    a2=[1;a2];
    z3=Theta2*a2;
    a3=sigmoid(z3);
    
    delt3=a3-y_tmp(:,ct1);
    %delt3=delt3(2:end,:);
    delt2=Theta2'*delt3.*(a2.*(1-a2));
    %delt2=Theta2'*delt3.*sigmoidGradient(z2);
    delt2=delt2(2:end,:);
    DELT2=DELT2+delt3*a2';
    DELT1=DELT1+delt2*a1;
end
Theta2_grad=DELT2/m;
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+lambda/m*Theta2(:,2:end);
Theta1_grad=DELT1/m;
Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+lambda/m*Theta1(:,2:end);
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
% freeze

end
