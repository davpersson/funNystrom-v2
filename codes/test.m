clc
clear

rng(0)
addpath('Other')
addpath('results')
tic
%--- PARAMETERS ---
steps = 7;
step_size = 1;

% --- ALLOCATE SPACE
errors_original = zeros(3,4,steps);
errors_function = errors_original;
errors_projection = errors_original;

%--- COLUMN SUBSET SELECTION ---
fprintf('Column subset selection\n')
% Matrix function
f = @(x) x./(x+1);

% Matrix
sigma = 1;
%kernel = @(x,y) exp(-norm(x-y,1)/sigma);
kernel = @(x,y) ((norm(x)/1000)^10 + (norm(y)/1000)^10)^(0.1);
data_matrix = orth(randn(1000,1000))*diag(1:1000);
A = build_kernel_matrix(data_matrix,kernel);
[U,S] = svd(A);
A = U*S*U';
fS = diag(f(diag(S)));
fA = U*fS*U';

% Parameters of test
k = 10;
q_list = k:step_size:(k+step_size*(steps-1));
[I,F] = subset_selection(A,max(q_list));
Q = eye(size(A)); Q = Q(:,I);

iteration = 0;
for r = q_list
    
    iteration = iteration + 1;
    [Uhat,Shat] = svd(F(:,1:r),'econ');%nystrom(A,Q(:,1:r));
    Shat = Shat^2;
    Uhat = Uhat(:,1:k); Shat = Shat(1:k,1:k);
    fShat = diag(f(diag(Shat)));
    
    [Uproj,Sproj,Vproj] = svd(Q(:,1:r)'*A,'econ');
    Uproj = Q(:,1:r)*Uproj;
    projAk = Uproj(:,1:k)*Sproj(1:k,1:k)*Vproj(:,1:k)';
    
    errors_projection(1,1,iteration) = max((sum(svd(A-projAk)))/trace(S((k+1):end,(k+1):end)) - 1,1e-16);
    errors_projection(1,2,iteration) = max((norm(A-projAk,'fro')/norm(S((k+1):end,(k+1):end),'fro')) - 1,1e-16);
    errors_projection(1,3,iteration) = max((norm(A-projAk)/S(k+1,k+1)) - 1,1e-16);
    errors_projection(1,4,iteration) = max(max(1-diag(Sproj(1:k,1:k))./diag(S(1:k,1:k))),1e-16);
    
    errors_original(1,1,iteration) = max(((trace(S) - trace(Shat))/trace(S((k+1):end,(k+1):end))) - 1,1e-16);
    errors_original(1,2,iteration) = max((norm(A-Uhat*Shat*Uhat','fro')/norm(S((k+1):end,(k+1):end),'fro')) - 1,1e-16);
    errors_original(1,3,iteration) = max((norm(A-Uhat*Shat*Uhat')/S(k+1,k+1)) - 1,1e-16);
    errors_original(1,4,iteration) = max(max(1-diag(Shat)./diag(S(1:k,1:k))),1e-16);
    
    errors_function(1,1,iteration) = max(((trace(fS) - trace(fShat))/trace(fS((k+1):end,(k+1):end))) - 1,1e-16);
    errors_function(1,2,iteration) = max((norm(fA-Uhat*fShat*Uhat','fro')/norm(fS((k+1):end,(k+1):end),'fro')) - 1,1e-16);
    errors_function(1,3,iteration) = max((norm(fA-Uhat*fShat*Uhat')/fS(k+1,k+1)) - 1,1e-16);
    errors_function(1,4,iteration) = max(max(1-diag(fShat)./diag(fS(1:k,1:k))),1e-16);
    
end

%--- KRYLOV ITERATION ---
fprintf('Krylov iteration\n')
% Matrix function
f = @(x) log(1+x);

% Matrix
n = 3000;
U = gallery('orthog',n);
S = diag((1:n).^(-1));
A = U*S*U';
fS = diag(f(diag(S)));
fA = U*fS*U';

% Parameters of test
k = 20;
q_list = step_size:step_size:(steps*step_size);
Q = block_lanczos(A,randn(n,k),max(q_list));

iteration = 0;
for q = q_list
    
    iteration = iteration + 1;
    [Uhat,Shat] = nystrom(A,Q(:,1:(k*q)));
    Uhat = Uhat(:,1:k); Shat = Shat(1:k,1:k);
    fShat = diag(f(diag(Shat)));
    
    [Uproj,Sproj,Vproj] = svd(Q(:,1:(k*q))'*A,'econ');
    Uproj = Q(:,1:(k*q))*Uproj;
    projAk = Uproj(:,1:k)*Sproj(1:k,1:k)*Vproj(:,1:k)';
    
    errors_projection(2,1,iteration) = max((sum(svd(A-projAk)))/trace(S((k+1):end,(k+1):end)) - 1,1e-16);
    errors_projection(2,2,iteration) = max((norm(A-projAk,'fro')/norm(S((k+1):end,(k+1):end),'fro')) - 1,1e-16);
    errors_projection(2,3,iteration) = max((norm(A-projAk)/S(k+1,k+1)) - 1,1e-16);
    errors_projection(2,4,iteration) = max(max(1-diag(Sproj(1:k,1:k))./diag(S(1:k,1:k))),1e-16);
    
    errors_original(2,1,iteration) = max(((trace(S) - trace(Shat))/trace(S((k+1):end,(k+1):end))) - 1,1e-16);
    errors_original(2,2,iteration) = max((norm(A-Uhat*Shat*Uhat','fro')/norm(S((k+1):end,(k+1):end),'fro')) - 1,1e-16);
    errors_original(2,3,iteration) = max((norm(A-Uhat*Shat*Uhat')/S(k+1,k+1)) - 1,1e-16);
    errors_original(2,4,iteration) = max(max(1-diag(Shat)./diag(S(1:k,1:k))),1e-16);
    
    errors_function(2,1,iteration) = max(((trace(fS) - trace(fShat))/trace(fS((k+1):end,(k+1):end))) - 1,1e-16);
    errors_function(2,2,iteration) = max((norm(fA-Uhat*fShat*Uhat','fro')/norm(fS((k+1):end,(k+1):end),'fro')) - 1,1e-16);
    errors_function(2,3,iteration) = max((norm(fA-Uhat*fShat*Uhat')/fS(k+1,k+1)) - 1,1e-16);
    errors_function(2,4,iteration) = max(max(1-diag(fShat)./diag(fS(1:k,1:k))),1e-16);
    
end

%--- POWER METHOD ---
fprintf('Power method\n')
% Matrix function
f = @(x) sqrt(x);

% Matrix
n = 3000;
U = gallery('orthog',n);
S = diag(exp(-(1:n)));
A = U*S*U';
fS = diag(f(diag(S)));
fA = U*fS*U';

% Parameters of test
k = 10;
q_list = step_size:step_size:(step_size*steps);
Q = orth(randn(n,k));

iteration = 0;
for q = q_list
    
    iteration = iteration + 1;
    [Uhat,Shat] = nystrom(A,Q);
    Uhat = Uhat(:,1:k); Shat = Shat(1:k,1:k);
    fShat = diag(f(diag(Shat)));
    
    [Uproj,Sproj,Vproj] = svd(Q'*A,'econ');
    Uproj = Q*Uproj;
    projAk = Uproj(:,1:k)*Sproj(1:k,1:k)*Vproj(:,1:k)';
    
    errors_projection(3,1,iteration) = max((sum(svd(A-projAk)))/trace(S((k+1):end,(k+1):end)) - 1,1e-16);
    errors_projection(3,2,iteration) = max((norm(A-projAk,'fro')/norm(S((k+1):end,(k+1):end),'fro')) - 1,1e-16);
    errors_projection(3,3,iteration) = max((norm(A-projAk)/S(k+1,k+1)) - 1,1e-16);
    errors_projection(3,4,iteration) = max(max(1-diag(Sproj(1:k,1:k))./diag(S(1:k,1:k))),1e-16);
    
    errors_original(3,1,iteration) = max(((trace(S) - trace(Shat))/trace(S((k+1):end,(k+1):end))) - 1,1e-16);
    errors_original(3,2,iteration) = max((norm(A-Uhat*Shat*Uhat','fro')/norm(S((k+1):end,(k+1):end),'fro')) - 1,1e-16);
    errors_original(3,3,iteration) = max((norm(A-Uhat*Shat*Uhat')/S(k+1,k+1)) - 1,1e-16);
    errors_original(3,4,iteration) = max(max(1-diag(Shat)./diag(S(1:k,1:k))),1e-16);
    
    errors_function(3,1,iteration) = max(((trace(fS) - trace(fShat))/trace(fS((k+1):end,(k+1):end))) - 1,1e-16);
    errors_function(3,2,iteration) = max((norm(fA-Uhat*fShat*Uhat','fro')/norm(fS((k+1):end,(k+1):end),'fro')) - 1,1e-16);
    errors_function(3,3,iteration) = max((norm(fA-Uhat*fShat*Uhat')/fS(k+1,k+1)) - 1,1e-16);
    errors_function(3,4,iteration) = max(max(1-diag(fShat)./diag(fS(1:k,1:k))),1e-16);
    
    Q = orth(A*Q);
    
end

fprintf('Finished\n')
toc

save('errors','errors_original','errors_function','errors_projection')
plotter('errors')

% %Run the experiment for all specified matrices
% for matrix = 1:4
%         
%     if matrix == 1
%         
%         %List of ranks of low-rank approximations
%         q_list = 1:10;
%         k = 10;
%         
%         %Determine the matrix function
%         fscalar = @(x) sqrt(x);
%         
%         % Determine method to compute orthonormal basis
%         method = 'power';
%         
%         %Specify A and f(A)
%         matrix_size = 5000;
%         parameter = 3;
%         U = gallery('orthog',matrix_size);
%         Lambda = diag((1:matrix_size).^(-parameter));
%         %fLambda = diag((1:matrix_size).^(-parameter/2));
%         
%         %Specify a filename
%         filename = 'results/algebraic3_sqrtm_power_k=10';
%         
%     elseif matrix == 2
%         
%         %List of ranks of low-rank approximations
%         q_list = 1:10;
%         k = 20;
%         
%         %Determine the matrix function
%         fscalar = @(x) log(1+x);
%         
%         % Determine method to compute orthonormal basis
%         method = 'krylov';
%         
%         %Specify A and f(A)
%         matrix_size = 5000;
%         parameter = 1;
%         U = gallery('orthog',matrix_size);
%         Lambda = diag((1:matrix_size).^(-parameter));
%         %fLambda = diag(fscalar(diag(Lambda)));
%         
%         %Specify a filename
%         filename = 'results/algebraic1_log_krylov_k=20';
%         
%     elseif matrix == 3
%        
%         %List of ranks of low-rank approximations
%         rank_list = 1:10;
%         k = 10;
%         
%         %Determine the matrix function
%         fscalar = @(x) x./(1+x);
%         
%         % Determine method to compute orthonormal basis
%         method = 'power';
%        
%         %Specify A and f(A)
%         matrix_size = 5000;
%         parameter = 10;
%         U = gallery('orthog',matrix_size);
%         Lambda = diag(exp(-(1:matrix_size)/parameter));
%         %fLambda = diag(fscalar(diag(Lambda)));
%         
%         %Specify a filename
%         filename = 'results/exponential10_inverse_power_k=10';
% 
%     elseif matrix == 4
%         
%        %List of ranks of low-rank approximations
%         rank_list = 1:10;
%         k = 10;
%         
%         %Determine the matrix function
%         fscalar = @(x) x./(1+x);
%         
%         % Determine method to compute orthonormal basis
%         method = 'krylov';
%        
%         %Specify A and f(A)
%         matrix_size = 5000;
%         parameter = 10;
%         U = gallery('orthog',matrix_size);
%         Lambda = diag(exp(-(1:matrix_size)/parameter));
%         %fLambda = diag(fscalar(diag(Lambda)));
%         
%         %Specify a filename
%         filename = 'results/exponential10_inverse_krylov_k=10';
%         
%     end
%     
%     matrix
%     
%     %Run test
%     test(U,Lambda,fscalar,q_list,k,method,filename)
%     
%     %Plot the results
%     plotter(filename);
%     
% end