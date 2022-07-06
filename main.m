%% Initialization
clear ; close all; clc
ddebug=0;
%% Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 28x28 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10
% (note that we have mapped "0" to label 10)

load('Train.mat');
y=cell2mat(Train(:,2))+1;
m = size(y, 1);%No. of samples
Xtmp=cell2mat(Train(:,1))';
X=reshape(Xtmp,input_layer_size,m)'/1000;%how does the array listed in ex4data1.mat?
clear Xtmp

load('Test.mat');
ytest=cell2mat(Test(:,2))+1;
mtest = size(ytest, 1);%No. of samples
Xtmp=cell2mat(Test(:,1))';
Xtest=reshape(Xtmp,input_layer_size,mtest)'/1000;%how does the array listed in ex4data1.mat?
clear Xtmp

if (0)
    % Randomly select 100 data points to display
    sel = randperm(size(X, 1));
    sel = sel(1:100);
    displayData(X(sel, :));
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end

if ddebug
    load('storedweights.mat');
else
    %Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    %Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
    %save('storedrandomweights.mat','Theta1','Theta2')
    load('storedrandomweights.mat');
end

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
%% Training NN
%  You have now implemented all the code necessary to train a neural
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%% gradient descent
num_iters=20;
alpha = 0.3;
J_ = zeros(num_iters, 1);
Xnew=[ones(m,1) X];

Theta1_pos_max=0;

for iter = 1:num_iters
    Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));
    
    z2=Theta1*Xnew';
    a2=sigmoid(z2);
    a2=[ones(1,m);a2];
    z3=Theta2*a2;
    a3=sigmoid(z3);
    
    J(iter)=sum(sum(-y_tmp.*log(a3)-(1-y_tmp).*log(1-a3)))/m;
    [iter,J(iter)];
    
    DELT2=0;
    DELT1=0;
    for ct1=1:m
        a1=Xnew(ct1,:);
        z2=Theta1*a1';
        a2=sigmoid(z2);
        a2=[1;a2];
        z3=Theta2*a2;
        a3=sigmoid(z3);
        
        delt3=a3-y_tmp(:,ct1);
        delt2=Theta2'*delt3.*(a2.*(1-a2));
        delt2=delt2(2:end,:);
        DELT2=DELT2+delt3*a2';
        DELT1=DELT1+delt2*a1;
    end
    Theta2_grad=DELT2/m;
    Theta2_grad(:,2:end)=Theta2_grad(:,2:end);
    Theta1_grad=DELT1/m;
    Theta1_grad(:,2:end)=Theta1_grad(:,2:end);
    Theta1=Theta1-alpha*Theta1_grad;
    Theta2=Theta2-alpha*Theta2_grad;
    
    %find max weight
    Theta1_pos_max=max(Theta1(Theta1>0));
    Theta1_neg_max=-max(-Theta1(Theta1<0));
    Theta2_pos_max=max(Theta2(Theta2>0));
    Theta2_neg_max=-max(-Theta2(Theta2<0));
end

if (0)
    %% Visualize Weights
    %  You can now "visualize" what the neural network is learning by
    %  displaying the hidden units to see what features they are capturing in
    %  the data.
    
    fprintf('\nVisualizing Neural Network... \n')
    
    displayData(Theta1(:, 2:end));
    
    fprintf('\nProgram paused. Press enter to continue.\n');
end
%% Implement Predict
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, Xtest);

fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == ytest)) * 100);
predic=mean(double(pred == ytest)) * 100;
iterplot=[1:num_iters];
