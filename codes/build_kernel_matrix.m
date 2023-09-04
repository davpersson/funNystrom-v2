function A = build_kernel_matrix(data_matrix,kernel)

n = size(data_matrix,2);
A = zeros(n,n);
for i = 1:n
    
    for j = 1:n
        
        A(i,j) = kernel(data_matrix(:,i),data_matrix(:,j));
        
    end
    
end

end