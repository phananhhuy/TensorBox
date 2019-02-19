function Z = convn(X,Y)
% Convolution between two order-N T-tensors X and Y
% 
% Phan Anh Huy, 2016
%
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
Z = ttensor(tensor(nkron(X.core,Y.core)),Z);