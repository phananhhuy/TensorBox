function Z = convn(X,Y)
% Convolution between two K-tensors or a K-tensor and a tensor X and Y
%
% Phan Anh Huy, 2016
%

if (isa(X,'double')||isa(X,'tensor'))
    Z = convn(Y,tensor(X));
    
elseif (isa(Y,'double')||isa(Y,'tensor'))
    
    Nx = ndims(X);Ny = ndims(Y);
    SzX = size(X);
    SzY = size(Y);
   
    R = size(X.u{1},2);
    
    Z = X.u;
    for n = 1:min(Nx,Ny)
        u1 = Z{n};
        u2 = eye(SzY(n));
        
        u2 = [zeros(SzX(n),SzY(n)); u2];
        u1 = [zeros(SzY(n),R); u1];
        
        u3 = ifft((khatrirao(fft(u1,[],1).', fft(u2,[],1).').'),[],1);
        u3 = u3(1:end-1,:);
        Z{n} = u3;
    end
    
    %
    if Nx<Ny
        for k = Nx+1:Ny
            Z{k} = speye(SzY(k));
        end
    end
    G = nkron(tendiag(ones(R,1),R*ones(1,min(Nx,Ny))),Y);
    Z = ttensor(tensor(G),Z);
    
elseif isa(X,'ktensor') && isa(Y,'ktensor')
    
    Nx = ndims(X);Ny = ndims(Y);
    if Nx<=Ny
        [X,Y] = deal(Y,X);
        [Nx,Ny] = deal(Ny,Nx);
    end
    R = size(X.u{1},2);
    S = size(Y.u{1},2);
    szX = size(X);
    szY = size(Y);
    
    Z = X.u;
    
    for k = 1:Ny
        u1 = X.u{k};
        u2 = Y.u{k};
        
        u1 = [zeros(szY(k),R) ; u1];
        u2 = [zeros(szX(k),S) ; u2];
        u3 = ifft((khatrirao(fft(u1,[],1).', fft(u2,[],1).').'),[],1);
        u3 = u3(1:end-1,:);
        Z{k} = u3;
    end
    
    for k = Ny+1:Nx
        Z{k} = kron(Z{k},ones(1,S));
    end
    Z = ktensor(kron(X.lambda,Y.lambda),Z);
    
end