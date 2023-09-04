function [I,F] = subset_selection(A,k)
I = zeros(k,1);
d = diag(A);
n = length(d);
F = zeros(n,k);

for i = 1:k
    
     I(i) = randsample(n,1,true,d);
     %g = A(:,I(i))-F(:,1:(i-1))*F(I(i),1:(i-1))';
     %F(:,i) = g/sqrt(g(I(i)));
     
     if i == 1
         
         F(:,1) = A(:,I(i))/sqrt(d(I(i)));
         
     else
         
         F(:,i) = (A(:,I(i)) - F(:,1:(i-1))*F(I(i),1:(i-1))')/sqrt(d(I(i)));
         
     end
     
     d = d - F(:,i).^2;
     d(d < 1e-14) = 0;
    
end
end