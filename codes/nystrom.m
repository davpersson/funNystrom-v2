function [U,S] = nystrom(A,Q)

%Implementation of the Nystrom approximation

%Compute matrix products with A
Y = A*Q;

%Regularizaton
[V,D,~] = svd(Q'*Y,'econ');
D(D < 5e-16*D(1,1)) = 0;
B = Y*(V*pinv(diag(sqrt(diag(D)))));
[U,Shat,~] = svd(B,'econ');
S = Shat^2;

end