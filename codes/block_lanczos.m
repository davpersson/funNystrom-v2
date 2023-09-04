function Q = block_lanczos(A,Z,q)

% Find block size
b = size(Z,2);

% First iteration
[Q,R0] = qr(Z,0);
R = R0;

for k = 0:q-1
    
    %Set Qk
    Qk = Q(:,(1+k*b):((k+1)*b));
    
    if k == 0
        
        Z = A*Qk;
        
    else
        
        % Set Qk-1
        Qkm1 = Q(:,(1+(k-1)*b):(k*b));
        
        Z = A*Qk - Qkm1*R';
        
    end
    
    % Obtain diagonal block in T
    M = Qk'*Z;
    
    if k == q-1
        
        return
        
    end
        
    
    % Reorthogonalization
    Z = Z - Qk*M;
    
    % Double reorthgonalization
    if k > 0
        
        Z = Z - Q(:,1:(k*b))*(Q(:,1:(k*b))'*Z);
        
    end
    
    % Obtain next block
    [Qkp1,R,P] = qr(Z,0);
    
    % Check if rank deficient
    if min(abs(diag(R))) <= (1e-10)*max(abs(diag(R)))
        
        % New block size
        b = max(find(abs(diag(R)) > (1e-10)*max(abs(diag(R)))));
        
        % Truncate
        R = R(:,1:b);
        Qkp1 = Qkp1(:,1:b);
        
    end
    
    % Permute back
    invP = zeros(1,length(P));
    invP(P) = 1:length(P);
    R = R(:,invP);
        
        
    Q = [Q Qkp1];
    
    
end

end