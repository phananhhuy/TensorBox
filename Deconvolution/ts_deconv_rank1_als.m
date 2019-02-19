function [H,U,output] = ts_deconv_rank1_als(Y,U,H,opts)
% ALS algorithm for rank-1 tensor deconvolution which is formulated as
%
%   X = H_1 * (a1_1 o a2_1 o ... o aN_1)  + ... + H_R * (a1_R o a2_R o ... o aN_R)
%
% Input:
%     Y:  tensor of size I1 x I2 x ... x IN
%     U:  cell array of initial Toeplitz matrices U{n} of size In x R,
%     U{n}(:,1,r) = a{n}_r
%     H:  (N+1) dimensional array of R pattern tensors of size J1 x J2 x
%     ... x JN x R
%
%
% Output
%    H : core tensors (or patterns) 
%    U : cell array of rank-1 activating tensors U(:,r), r = r, ..., R
%
% REF:
%
% [1] A. -H. Phan , P. Tichavsk?, and A. Cichocki, ?Low rank tensor
% deconvolution?, in IEEE International Conference on Acoustics, Speech and
% Signal Processing (ICASSP), pp. 2169 - 2173, 2015.
%
%
% Phan Anh Huy, August, 2014
%
% TENSOR BOX, v.2015

if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    H = param; return
end

normY = norm(Y);normY2 = normY^2;
% Y = tensor(Y);

%% Initialization 
% [H,U] = ts_deconv_init(Y,R,SzH,opts);

% Y = tensor(permute(double(Y),[4 1 2 3]));

%% Check size and permute data 

SzU = cell2mat(cellfun(@size,U,'uni',0)); % [SzY SzH R]
%SzH = size(H); % [k x SzH x R]
SzY = size(Y);

if  all(SzY(1:size(SzU,1))' == SzU(:,1))
    % Permute Y to be of size (K x I1 x I2 x I3)
    Y = tensor(permute(double(Y),[4 1 2 3]));

elseif any(SzY(2:end)' ~= SzU(:,1))
    error('Initial values do not match tensor sizes.')
end
N = ndims(Y)-1;

%%
 
error_vec = [];
if param.printitn ~=0
    fprintf('Iter   |  Error     |  delta \n');
    %     fprintf('%-4d   |  %.6f  |   \n',0,approx_error);
end


for iter = 1:param.maxiters
    regterm = 0;
    for n = 1:N
        if ismember(n,param.sparse)
            %             [U{n},tau,H] = update_Ucompact_sparse(Y,U,H,n,normY2); % sparse Un
            [U{n},tau,H,Vvec] = update_Ucompact_sparse2(Y,U,H,n,normY2); % sparse Un
            regterm = regterm + tau*sum(sum(abs(U{n}(:,1,:))));
            
        elseif ismember(n,param.orthogonal)
            [Un,errornew,Vvec] = Update_Uortho2(Y,U,H,n,normY2);
            errornew = errornew^2;
            if (iter < 10) || (errornew<= approx_error)
                approx_error = errornew;
                U{n} = Un;
            end
            %             Vvec = [];
        else
            if ((size(U{n},1)-size(U{n},3))*size(U{n},3)) < 10000
                if size(U{n},2) > 1
                    [U{n},Vvec] = Update_Ucompact2(Y,U,H,n); % sequential update ur
                    %                     %  [U{n},Vvec] = Update_Ucompact(Y,U,H,n);
                else
                    [U{n},Vvec] = Update_Ucompact_J1(Y,U,H,n); % Update when L_n = 1
                end
            else
                [U{n},foe,Vvec] = hUpdate_Ucompact(Y,U,H,n); % sequential update ur
            end
        end
        
        if ismember(N+1,param.sparse)
            [H,approx_error] = update_Hsparse(Y,U,H,normY2,Vvec);
        else
            %             condU = zeros(R,1);
            %             for r = 1:R
            %                 condU(r) = cond(U{n}(:,:,r));
            %             end
            %             if (numel(H) <= 1000) || (any(condU>1e5))
            [H,approx_error] = update_H(Y,U,Vvec);
            %             else
            %                 [H,approx_error] = hupdate_H(Y,U,H,Vvec);
            %             end
        end
        approx_error =  ((normY2 + approx_error)/2 + regterm)/normY2; % fast computation
        
    end
    
    
    % Approximation approx_error
    %approx_error = normFrob(Y,U,H,normY2)/normY;
    error_vec = [error_vec  approx_error];
    if (iter>1) && (mod(iter,param.printitn)==0 )
        fprintf('%-4d   |  %.6f  |  %.6d \n',iter,approx_error,error_vec(iter)-error_vec(iter-1));
    end
    
    if (iter>1) && (abs(error_vec(iter)-error_vec(iter-1))< param.tol)
        break;
    end
end

output = struct('Error',error_vec,'NoIters',numel(error_vec));
end


%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init','cpd',@(x) (iscell(x) || isa(x,'ktensor')||...
    ismember(x(1:3),{'cpd'}) || ((numel(x)>3) && ismember(x(1:4),{'rand' 'tedi'}) )));
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('sparse',0);% 0 implies loading componetns and patterns Hr are without sparsity constraints.
% [1 2 4] means the first two factors and cores are imposed
% sparsity constraints.

param.addOptional('orthogonal',0); % for orthogonality constraints
% 0 implies loading componetns and patterns Hr are without orthogonality constraints.


param.parse(opts);
param = param.Results;

end


%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [Un,H,Vvec] = hUpdate_Ucompact(Y,U,H,mode)
% U:  cellarray of N factor matrices U{n} of size In x J1 x R, whose each
% slice U{n}(:,:,r) is Toeplitz matrix of the loading component u{n}_r (In)
SzH = size(H); % Lt x J1 x ...x JN x R
SzY = size(Y); % Lt x I1 x ...x IN
SzU = [SzY(2:end)' SzH(2:end-1)'];
R = SzH(end);
L = SzU(mode,2);K = SzU(mode,1)-L+1;
Y = double(Y);

N = numel(U);%Y = tensor(Y);

% % Toeplitz operator for a Toeplitz matrix of size (J*L) x (J+L-1)
% ri = reshape(1:L*SzU(mode,1),L,SzU(mode,1));ri =  ri(:);
% ci = bsxfun(@plus,(L:-1:1)',0:SzU(mode,1)-1)'; ci = ci(:);
% T_KJ = sparse(ri,ci,ones(L*SzU(mode,1),1),SzU(mode,1)*L,SzU(mode,1)+L-1);
% T_KJ = T_KJ(:,L:SzU(mode,1));
%
% % Toeplitz operator for banded Toeplitz matrix of size K x K from vectors phi Lx1
% nnzK = (2*L-1)*K - L*(L-1);
%
% dix = (K+1)*ones(nnzK-1,1);
% ix3 = cumsum(K-L+1:K-2);
% dix(ix3) = (K+1)*(L-1:-1:2);
% dix(end-ix3+1) = (K+1)*(L-1:-1:2);
% ix = cumsum([L;dix]);
% [ir,ic] = ind2sub([K^2,2*L-1],ix);
% T_KK = sparse(ir,ic,ones(numel(ir),1),K^2,2*L-1);
%
% % irc = find(sum(T_KK,2));
% % [rows0,cols0] = ind2sub([K K],irc);
% % [rows0,cols0] = find(reshape(sum(T_KK,2),[K K]));


% Toeplitz operator for a Toeplitz matrix of size (J*L) x (J+L-1)
% ri = reshape(1:L*SzU(mode,1),L,SzU(mode,1));ri =  ri(:);
% ci = bsxfun(@plus,(L:-1:1)',0:SzU(mode,1)-1)'; ci = ci(:);
% T_KJ = sparse(ri,ci,ones(L*SzU(mode,1),1),SzU(mode,1)*L,SzU(mode,1)+L-1);
% T_KJ = T_KJ(:,L:SzU(mode,1));

cols = repmat((1:K)',L,1);
rows = bsxfun(@plus,(1:K)',0:SzU(mode,1)+1:(SzU(mode,1)+1)*(L-1));
T_KJ = sparse(rows(:),cols(:),ones(numel(rows),1),SzU(mode,1)*L,K);

% % Toeplitz operator to generate an LxL Toeplitz matrix from a vector of lenght (2L+1)
% % vec(X) = T_LL * x
% % Note that T_LL is used to sum diagonals of a matrix X of size LxL:  T_LL * vec(X)
% ri = reshape(1:L*L,L,L);ri =  ri(:);
% ci = bsxfun(@plus,(L:-1:1)',0:L-1)'; ci = ci(:);
% T_LL = sparse(ri,ci,ones(L*L,1),L*L,L+L-1)';

% Toeplitz operator to generate a banded Toeplitz matrix of size K x K from vectors phi Lx1
nnzK = (2*L-1)*K - L*(L-1);
dix = (K+1)*ones(nnzK-1,1);
ix3 = cumsum(K-L+1:K-2);
dix(ix3) = (K+1)*(L-1:-1:2);
dix(end-ix3+1) = (K+1)*(L-1:-1:2);
ix = cumsum([L;dix]);
[ir,ic] = ind2sub([K^2,2*L-1],ix);
T_KK = sparse(ir,ic,ones(numel(ir),1),K^2,2*L-1);

% % This vector is used to replicate entries of vector phi of lenght L in
% % vectorization of the KxK Toeplitz matrix,
% % i.e. phi(phimap) = nonzero(Phi)
%
% phimap = nonzeros(T_KK*[1:2*L-1]');
% [rows0,cols0] = find(reshape(sum(T_KK,2),[K K]));
%
% % For a Toeplitz matrix Phi of size KxK generated from vector phi of length L,
% % Phi is a sparse matrix defined as
% % Phi = sparse(rows0,cols0,phi(phimap),K,K);


krondim = [1:mode-1 mode+1:N];

LT = size(Y,1);
if nargout > 1
    W = zeros(SzU(mode,1)*R,LT*prod(SzU([1:mode-1 mode+1:end],2)));
end


Hm = tenmat(H,mode+1);
Hm = reshape(Hm.data,SzU(mode,2), [], R);

for r = 1:R
    % Estimate component u_r
    Ur = cellfun(@(x) x(:,:,r),U,'uni',0);
    wr = mttm(Y,Ur,mode+1);
    
    wr = reshape(double(permute(wr,[mode+1 1:mode mode+2:N+1])),SzU(mode,1),[]);
    
    if nargout > 2
        % The following is for update core tensor H_r
        W(SzU(mode)*(r-1)+1:SzU(mode)*r,:) = wr; % Yx {U_r}
    end
    
    wr = wr * Hm(:,:,r)';
    wr = T_KJ'*wr(:); % take sum along diagonals
    
    for s = [1:r-1 r+1:R r]
        
        UtU = U{krondim(1)}(:,:,r)'*U{krondim(1)}(:,:,s);
        for n = krondim(2:end)
            UtU = kron(U{n}(:,:,r)'*U{n}(:,:,s),UtU);
        end
        Zrs = Hm(:,:,r) * UtU  * Hm(:,:,s)'; % ttt(ttensor(Hr,Ur),ttensor(Hs,Us),-mode)
        
        phi = zeros(2*size(Zrs,1)-1,1);
        for j = -size(Zrs,1)+1:size(Zrs,1)-1
            phi(j+size(Zrs,1)) = sum(diag(Zrs,j));
        end
        
        if s ~= r
            Phirs = reshape(T_KK*phi,K,K)';
            
            %             %Phirs = toeplitz([phi(L:end); zeros(SzU(mode,1)-2*L+1,1)],...
            %             %    [phi(L:-1:1)' zeros(1,SzU(mode,1)-2*L+1)]);
            %             Phirs = smtoep([phi(L:end); zeros(SzU(mode,1)-2*L+1,1)],...
            %                  [phi(L:-1:1)' zeros(1,SzU(mode,1)-2*L+1)]);
            wr = wr - Phirs * U{mode}(1:SzU(mode,1)-L+1,1,s);
        end
    end
    
    % Update in equation (14)
    Phirr = reshape(T_KK*phi,K,K);
    ur = Phirr\wr;
    
    ur = ur/norm(ur);
    U{mode}(:,:,r) = convmtx(ur,L);
end


Un = U{mode};

if nargout > 2
    Vvec = zeros(SzU(mode,2)*R,LT*prod(SzU([1:mode-1 mode+1:end],2)));
    
    for r = 1:R
        Vvec(SzU(mode,2)*(r-1)+1:SzU(mode,2)*r,:) = Un(:,:,r)'*W(SzU(mode,1)*(r-1)+1:SzU(mode,1)*r,:);
    end
    
    Vvec = reshape(Vvec,[SzU(mode,2),R,SzU([1:mode-1 mode+1:end],2)',LT]);
    Vvec = permute(Vvec,[3:mode+1 1 mode+2:N+1 2 N+2]);
    Vvec = reshape(Vvec,[],LT);
end

end


%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [Un,Vvec] = Update_Ucompact2(Y,U,H,mode)
% U:  cellarray of N factor matrices U{n} of size In x J1 x R, whose each
% slice U{n}(:,:,r) is Toeplitz matrix of the loading component u{n}_r (In)
% Vvec : Yx{Ur}

SzH = size(H); % Lt x J1 x ...x JN x R
SzY = size(Y); % Lt x I1 x ...x IN
if ndims(H) == ndims(Y)
    SzH(end+1) = 1; % means that R == 1
end

if ndims(H) < ndims(Y)
    SzH(end+1:ndims(Y)+1) = 1; % means that R == 1
end

SzU = [SzY(2:end)' SzH(2:end-1)'];
R = SzH(end);
L = SzU(mode,2);K = SzU(mode,1)-L+1;
Y = double(Y);
N = numel(U);%Y = tensor(Y);

% Toeplitz operator for a Toeplitz matrix of size (J*L) x (J+L-1)
% ri = reshape(1:L*SzU(mode,1),L,SzU(mode,1));ri =  ri(:);
% ci = bsxfun(@plus,(L:-1:1)',0:SzU(mode,1)-1)'; ci = ci(:);
% T_KJ = sparse(ri,ci,ones(L*SzU(mode,1),1),SzU(mode,1)*L,SzU(mode,1)+L-1);
% T_KJ = T_KJ(:,L:SzU(mode,1));

cols = repmat((1:K)',L,1);
rows = bsxfun(@plus,(1:K)',0:SzU(mode,1)+1:(SzU(mode,1)+1)*(L-1));
T_KJ = sparse(rows(:),cols(:),ones(numel(rows),1),SzU(mode,1)*L,K);

% Toeplitz operator to generate an LxL Toeplitz matrix from a vector of lenght (2L+1)
% vec(X) = T_LL * x
% Note that T_LL is used to sum diagonals of a matrix X of size LxL:  T_LL * vec(X)
ri = reshape(1:L*L,L,L);ri =  ri(:);
ci = bsxfun(@plus,(L:-1:1)',0:L-1)'; ci = ci(:);
T_LL = sparse(ri,ci,ones(L*L,1),L*L,L+L-1)';

% Toeplitz operator to generate a banded Toeplitz matrix of size K x K from vectors phi Lx1
nnzK = (2*L-1)*K - L*(L-1);
dix = (K+1)*ones(nnzK-1,1);
ix3 = cumsum(K-L+1:K-2);
dix(ix3) = (K+1)*(L-1:-1:2);
dix(end-ix3+1) = (K+1)*(L-1:-1:2);
ix = cumsum([L;dix]);
[ir,ic] = ind2sub([K^2,2*L-1],ix);
T_KK = sparse(ir,ic,ones(numel(ir),1),K^2,2*L-1);

% This vector is used to replicate entries of vector phi of lenght L in
% vectorization of the KxK Toeplitz matrix,
% i.e. phi(phimap) = nonzero(Phi)

phimap = nonzeros(T_KK*[1:2*L-1]');
[rows0,cols0] = find(reshape(sum(T_KK,2),[K K]));


% For a Toeplitz matrix Phi of size KxK generated from vector phi of length L,
% Phi is a sparse matrix defined as
% Phi = sparse(rows0,cols0,phi(phimap),K,K);

krondim = [1:mode-1 mode+1:N]; % modes involving in Kronecker products of Ur^T Us

% Prepare the common computation in estimation Un and core tensors H_r
LT = size(Y,1); % this parameter is often of 1, unless the data is converted to Toeplitz structure.
if nargout > 1
    W = zeros(SzU(mode,1)*R,LT*prod(SzU([1:mode-1 mode+1:end],2)));
end

Vvec = zeros(SzY(mode+1)*L,R);

Hm = tenmat(H,mode+1);
Hm = reshape(Hm.data,SzU(mode,2), [], R);

rcv = ones(R^2*nnzK,3); % matrix of [rows cols values] of nonzero entries of the matrix Phi
cnt = 0;
for r = 1:R
    for s = r:R
        
        temp = U{krondim(1)}(:,:,r)'*U{krondim(1)}(:,:,s);
        for n = krondim(2:end)
            temp = kron(U{n}(:,:,r)'*U{n}(:,:,s),temp);
        end
        temp = Hm(:,:,r) * temp  * Hm(:,:,s)'; % ttt(ttensor(Hr,Ur),ttensor(Hs,Us),-mode)
        
        cnt = cnt+1;
        phi = T_LL*temp(:); % sums along diagonal of matrix temp
        
        %phi2 = T_KK*phi;
        %vals2 = nonzeros(phi2);
        vals = phi(phimap);
        
        rows = rows0+K*(r-1);
        cols = cols0+K*(s-1);
        rcv(nnzK*(cnt-1)+1:nnzK*cnt,:) = [rows cols vals];
        if s~=r
            cnt = cnt+1;
            rcv(nnzK*(cnt-1)+1:nnzK*cnt,:) = [cols rows vals];
        end
    end
    
    Ur = cellfun(@(x) x(:,:,r),U,'uni',0);
    %     zy = ttm(Y,Ur(krondim2),krondim2+1,'t');
    zy = mttm(Y,Ur,mode+1);
    zy = tenmat(zy,mode+1);
    
    if nargout > 1
        % The following is for update core tensor H_r
        W(SzU(mode)*(r-1)+1:SzU(mode)*r,:) = zy; % Yx {U_r}
    end
    
    zy = zy * Hm(:,:,r)';
    Vvec(:,r) = zy(:);
end

w = T_KJ'*Vvec;


% Generate sparse symmetric matrix Phi of size KR x KR
% rcv(rcv(:,1) == 0,:) =[];
% Phi = sparse(rcv(:,1),rcv(:,2),rcv(:,3),K*R,K*R);
%
% % Update new un = [u1;u2;...;uR] following equation (13)
% un = (Phi+1e-12*speye(size(Phi)))\w(:); % This is much faster than the Levinson method

%  Rearrange Phi matrix into block Toeplit matrix and inverse using the Levinson block Toeplitz method
%pp = bsxfun(@plus,1:K,(0:K:K*(R-1))');pp = pp(:);%[foe,pp_i] = sort(pp);
pp = reshape(1:R*K,K,[])'; pp = pp(:);
Phi = sparse(rcv(:,1),rcv(:,2),rcv(:,3),K*R,K*R);
un = Phi(pp,pp)\w(pp); %faster than using the Levinson inverse method
% un = block_levinson(w(pp), full(Phi(:,1:R)));
un(pp) = un;

un = reshape(un, [], R);
un = bsxfun(@rdivide,un,sqrt(sum(un.^2)));
Un = zeros(SzU(mode,1),L,R);
for r = 1:R
    Un(:,:,r) = convmtx(un(:,r),L); % COnvert un to Toeplitz matrix Un % this step may be redundant
end


% Return the vector product Y x {Ur} for update Hn
if nargout > 1
    Vvec = zeros(SzU(mode,2)*R,LT*prod(SzU([1:mode-1 mode+1:end],2)));
    
    for r = 1:R
        Vvec(SzU(mode,2)*(r-1)+1:SzU(mode,2)*r,:) = Un(:,:,r)'*W(SzU(mode,1)*(r-1)+1:SzU(mode,1)*r,:);
    end
    
    Vvec = reshape(Vvec,[SzU(mode,2),R,SzU([1:mode-1 mode+1:end],2)',LT]);
    Vvec = permute(Vvec,[3:mode+1 1 mode+2:N+1 2 N+2]);
    Vvec = reshape(Vvec,[],LT);
end
end


%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [Un,Vvec] = Update_Ucompact_J1(Y,U,H,mode)
% THis update runs when J = 1
% U:  cellarray of N factor matrices U{n} of size In x J1 x R, whose each
% slice U{n}(:,:,r) is Toeplitz matrix of the loading component u{n}_r (In)
% Vvec : Yx{Ur}

SzH = size(H); % Lt x J1 x ...x JN x R
SzY = size(Y); % Lt x I1 x ...x IN
if ndims(H) == ndims(Y)
    SzH(end+1) = 1; % means that R == 1
end

if ndims(H) < ndims(Y)
    SzH(end+1:ndims(Y)+1) = 1; % means that R == 1,
end

SzU = [SzY(2:end)' SzH(2:end-1)'];
R = SzH(end);
L = SzU(mode,2);K = SzU(mode,1)-L+1;
Y = double(Y);
N = numel(U);%Y = tensor(Y);

krondim = [1:mode-1 mode+1:N]; % modes involving in Kronecker products of Ur^T Us

% Prepare the common computation in estimation Un and core tensors H_r
LT = size(Y,1); % this parameter is often of 1, unless the data is converted to Toeplitz structure.
if nargout > 1
    W = zeros(SzU(mode,1)*R,LT*prod(SzU([1:mode-1 mode+1:end],2)));
end

Vvec = zeros(SzY(mode+1)*L,R);

Hm = tenmat(tensor(H,SzH),mode+1);
Hm = reshape(Hm.data,SzU(mode,2), [], R);

Phi = zeros(R,R);
for r = 1:R
    for s = r:R
        
        temp = U{krondim(1)}(:,:,r)'*U{krondim(1)}(:,:,s);
        for n = krondim(2:end)
            temp = kron(U{n}(:,:,r)'*U{n}(:,:,s),temp);
        end
        temp = Hm(:,:,r) * temp  * Hm(:,:,s)'; % ttt(ttensor(Hr,Ur),ttensor(Hs,Us),-mode)
        
        Phi(r,s) = temp;
        if s~=r
            Phi(s,r) = temp;
        end
    end
    
    Ur = cellfun(@(x) x(:,:,r),U,'uni',0);
    %     zy = ttm(Y,Ur(krondim),krondim+1,'t');
    zy = mttm(Y,Ur,mode+1);
    zy = tenmat(zy,mode+1);
    
    if nargout > 1
        % The following is for update core tensor H_r
        W(SzU(mode)*(r-1)+1:SzU(mode)*r,:) = zy; % Yx {U_r}
    end
    
    zy = zy * Hm(:,:,r)';
    Vvec(:,r) = zy(:);
end

%  Rearrange Phi matrix into block Toeplit matrix and inverse using the Levinson block Toeplitz method
un = Vvec/Phi;
un = bsxfun(@rdivide,un,sqrt(sum(un.^2)));
Un = zeros(SzU(mode,1),L,R);
for r = 1:R
    Un(:,:,r) = convmtx(un(:,r),L); % COnvert un to Toeplitz matrix Un % this step may be redundant
end


% Return the vector product Y x {Ur} for update Hn
if nargout > 1
    Vvec = zeros(SzU(mode,2)*R,LT*prod(SzU([1:mode-1 mode+1:end],2)));
    
    for r = 1:R
        Vvec(SzU(mode,2)*(r-1)+1:SzU(mode,2)*r,:) = Un(:,:,r)'*W(SzU(mode,1)*(r-1)+1:SzU(mode,1)*r,:);
    end
    
    Vvec = reshape(Vvec,[SzU(mode,2),R,SzU([1:mode-1 mode+1:end],2)',LT]);
    Vvec = permute(Vvec,[3:mode+1 1 mode+2:N+1 2 N+2]);
    Vvec = reshape(Vvec,[],LT);
end
end

%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [Un,Vvec] = Update_Ucompact2b(Y,U,H,mode)
% U:  cellarray of N factor matrices U{n} of size In x J1 x R, whose each
% slice U{n}(:,:,r) is Toeplitz matrix of the loading component u{n}_r (In)
% Vvec : Yx{Ur}

persistent Phi Kb Rb Lb T_KJ T_LL T_KK nnzK idxrc;


SzH = size(H); % Lt x J1 x ...x JN x R
SzY = size(Y); % Lt x I1 x ...x IN
SzU = [SzY(2:end)' SzH(2:end-1)'];
R = SzH(end);
L = SzU(mode,2);K = SzU(mode,1)-L+1;
Y = double(Y);
N = numel(U);%Y = tensor(Y);

if isempty(Kb) || (K~=Kb) || (Rb ~= R) || (Lb~= L)
    Kb = K; Rb = R; Lb = L;
    % Toeplitz operator for a Toeplitz matrix of size (J*L) x (J+L-1)
    % ri = reshape(1:L*SzU(mode,1),L,SzU(mode,1));ri =  ri(:);
    % ci = bsxfun(@plus,(L:-1:1)',0:SzU(mode,1)-1)'; ci = ci(:);
    % T_KJ = sparse(ri,ci,ones(L*SzU(mode,1),1),SzU(mode,1)*L,SzU(mode,1)+L-1);
    % T_KJ = T_KJ(:,L:SzU(mode,1));
    
    cols = repmat((1:K)',L,1);
    rows = bsxfun(@plus,(1:K)',0:SzU(mode,1)+1:(SzU(mode,1)+1)*(L-1));
    T_KJ = sparse(rows(:),cols(:),ones(numel(rows),1),SzU(mode,1)*L,K);
    
    % Toeplitz operator to generate an LxL Toeplitz matrix from a vector of lenght (2L+1)
    % vec(X) = T_LL * x
    % Note that T_LL is used to sum diagonals of a matrix X of size LxL:  T_LL * vec(X)
    ri = reshape(1:L*L,L,L);ri =  ri(:);
    ci = bsxfun(@plus,(L:-1:1)',0:L-1)'; ci = ci(:);
    T_LL = sparse(ri,ci,ones(L*L,1),L*L,L+L-1)';
    
    % Toeplitz operator to generate a banded Toeplitz matrix of size K x K from vectors phi Lx1
    nnzK = (2*L-1)*K - L*(L-1);
    dix = (K+1)*ones(nnzK-1,1);
    ix3 = cumsum(K-L+1:K-2);
    dix(ix3) = (K+1)*(L-1:-1:2);
    dix(end-ix3+1) = (K+1)*(L-1:-1:2);
    ix = cumsum([L;dix]);
    [ir,ic] = ind2sub([K^2,2*L-1],ix);
    T_KK = sparse(ir,ic,ones(numel(ir),1),K^2,2*L-1);
    
    % This vector is used to replicate entries of vector phi of lenght L in
    % vectorization of the KxK Toeplitz matrix,
    % i.e. phi(phimap) = nonzero(Phi)
    
    phimap = nonzeros(T_KK*[1:2*L-1]');
    [rows0,cols0] = find(reshape(sum(T_KK,2),[K K]));
    
    
    % For a Toeplitz matrix Phi of size KxK generated from vector phi of length L,
    % Phi is a sparse matrix defined as
    % Phi = sparse(rows0,cols0,phi(phimap),K,K);
    
    
    rcv = ones(R^2*nnzK,3); % matrix of [rows cols values] of nonzero entries of the matrix Phi
    cnt = 0;
    for r = 1:R
        for s = r:R
            cnt = cnt+1;
            rows = rows0+K*(r-1);
            cols = cols0+K*(s-1);
            rcv(nnzK*(cnt-1)+1:nnzK*cnt,1:2) = [rows cols];
            
            if s~=r
                cnt = cnt+1;
                rcv(nnzK*(cnt-1)+1:nnzK*cnt,1:2) = [cols rows];
            end
        end
    end
    Phi = sparse(rcv(:,1),rcv(:,2),rcv(:,3),K*R,K*R);
    idxrc = sub2ind([K*R,K*R],rcv(:,1),rcv(:,2));
end

krondim = [1:mode-1 mode+1:N]; % modes involving in Kronecker products of Ur^T Us

% Prepare the common computation in estimation Un and core tensors H_r
LT = size(Y,1); % this parameter is often of 1, unless the data is converted to Toeplitz structure.
if nargout > 1
    W = zeros(SzU(mode,1)*R,LT*prod(SzU([1:mode-1 mode+1:end],2)));
end

Vvec = zeros(SzY(mode+1)*L,R);

Hm = tenmat(H,mode+1);
Hm = reshape(Hm.data,SzU(mode,2), [], R);


cnt = 0;
for r = 1:R
    for s = r:R
        
        temp = U{krondim(1)}(:,:,r)'*U{krondim(1)}(:,:,s);
        for n = krondim(2:end)
            temp = kron(U{n}(:,:,r)'*U{n}(:,:,s),temp);
        end
        temp = Hm(:,:,r) * temp  * Hm(:,:,s)'; % ttt(ttensor(Hr,Ur),ttensor(Hs,Us),-mode)
        
        cnt = cnt+1;
        phi = T_LL*temp(:); % sums along diagonal of matrix temp
        
        %phi2 = T_KK*phi;
        %vals2 = nonzeros(phi2);
        vals = phi(phimap);
        
        %         rows = rows0+K*(r-1);
        %         cols = cols0+K*(s-1);
        %         rcv(nnzK*(cnt-1)+1:nnzK*cnt,:) = [rows cols vals];
        %         if s~=r
        %             cnt = cnt+1;
        %             rcv(nnzK*(cnt-1)+1:nnzK*cnt,:) = [cols rows vals];
        %         end
        
        rcv(nnzK*(cnt-1)+1:nnzK*cnt,3) = vals;
        if s~=r
            cnt = cnt+1;
            rcv(nnzK*(cnt-1)+1:nnzK*cnt,3) = vals;
        end
        
        %         temp = toeplitz([phi(L:-1:1); zeros(SzU(mode,1)-2*L+1,1)],[phi(L:end)' zeros(1,SzU(mode,1)-2*L+1)]);
        %         Phi2(K*(r-1)+1:K*r,K*(s-1)+1:K*s) = sparse(temp);
        %         if s~=r
        %             Phi2(K*(s-1)+1:K*s,K*(r-1)+1:K*r) = Phi2(K*(r-1)+1:K*r,K*(s-1)+1:K*s)';
        %         end
    end
    
    Ur = cellfun(@(x) x(:,:,r),U,'uni',0);
    %     zy = ttm(Y,Ur(krondim2),krondim2+1,'t');
    zy = mttm(Y,Ur,mode+1);
    zy = tenmat(zy,mode+1);
    
    if nargout > 1
        % The following is for update core tensor H_r
        W(SzU(mode)*(r-1)+1:SzU(mode)*r,:) = zy; % Yx {U_r}
    end
    
    zy = zy * Hm(:,:,r)';
    Vvec(:,r) = zy(:);
end

w = T_KJ'*Vvec;


% Generate sparse symmetric matrix Phi of size KR x KR
% rcv(rcv(:,1) == 0,:) =[];
% Phi = sparse(rcv(:,1),rcv(:,2),rcv(:,3),K*R,K*R);
%
% % Update new un = [u1;u2;...;uR] following equation (13)
% un = (Phi+1e-12*speye(size(Phi)))\w(:); % This is much faster than the Levinson method

%  Rearrange Phi matrix into block Toeplit matrix and inverse using the Levinson block Toeplitz method
pp = reshape(1:R*K,K,[])'; pp = pp(:);
%Phi = sparse(rcv(:,1),rcv(:,2),rcv(:,3),K*R,K*R);
Phi(idxrc) = rcv(:,3);
un = Phi(pp,pp)\w(pp); %faster than using the Levinson inverse method
% un = block_levinson(w(pp), full(Phi(:,1:R)));
un(pp) = un;

un = reshape(un, [], R);
un = bsxfun(@rdivide,un,sqrt(sum(un.^2)));
Un = zeros(SzU(mode,1),L,R);
for r = 1:R
    Un(:,:,r) = convmtx(un(:,r),L); % COnvert un to Toeplitz matrix Un % this step may be redundant
end


% Return the vector product Y x {Ur} for update Hn
if nargout > 1
    Vvec = zeros(SzU(mode,2)*R,LT*prod(SzU([1:mode-1 mode+1:end],2)));
    
    for r = 1:R
        Vvec(SzU(mode,2)*(r-1)+1:SzU(mode,2)*r,:) = Un(:,:,r)'*W(SzU(mode,1)*(r-1)+1:SzU(mode,1)*r,:);
    end
    
    Vvec = reshape(Vvec,[SzU(mode,2),R,SzU([1:mode-1 mode+1:end],2)',LT]);
    Vvec = permute(Vvec,[3:mode+1 1 mode+2:N+1 2 N+2]);
    Vvec = reshape(Vvec,[],LT);
end
end



%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [Un,tau,H,Vvec] = update_Ucompact_sparse2(Y,U,H,mode,normY2)

% U:  cellarray of N factor matrices U{n} of size In x J1 x R, whose each
% slice U{n}(:,:,r) is Toeplitz matrix of the loading component u{n}_r (In)
SzH = size(H); % Lt x J1 x ...x JN x R
SzY = size(Y); % Lt x I1 x ...x IN
SzU = [SzY(2:end)' SzH(2:end-1)'];
R = SzH(end);
L = SzU(mode,2);K = SzU(mode,1)-L+1;
Y = double(Y);
N = numel(U);%Y = tensor(Y);


% Toeplitz operator for a Toeplitz matrix of size (J*L) x (J+L-1)
% ri = reshape(1:L*SzU(mode,1),L,SzU(mode,1));ri =  ri(:);
% ci = bsxfun(@plus,(L:-1:1)',0:SzU(mode,1)-1)'; ci = ci(:);
% T_KJ = sparse(ri,ci,ones(L*SzU(mode,1),1),SzU(mode,1)*L,SzU(mode,1)+L-1);
% T_KJ = T_KJ(:,L:SzU(mode,1));

cols = repmat((1:K)',L,1);
rows = bsxfun(@plus,(1:K)',0:SzU(mode,1)+1:(SzU(mode,1)+1)*(L-1));
T_KJ = sparse(rows(:),cols(:),ones(numel(rows),1),SzU(mode,1)*L,K);

% Toeplitz operator to generate an LxL Toeplitz matrix from a vector of lenght (2L+1)
% vec(X) = T_LL * x
% Note that T_LL is used to sum diagonals of a matrix X of size LxL:  T_LL * vec(X)
ri = reshape(1:L*L,L,L);ri =  ri(:);
ci = bsxfun(@plus,(L:-1:1)',0:L-1)'; ci = ci(:);
T_LL = sparse(ri,ci,ones(L*L,1),L*L,L+L-1)';

% Toeplitz operator to generate a banded Toeplitz matrix of size K x K from vectors phi Lx1
nnzK = (2*L-1)*K - L*(L-1);
dix = (K+1)*ones(nnzK-1,1);
ix3 = cumsum(K-L+1:K-2);
dix(ix3) = (K+1)*(L-1:-1:2);
dix(end-ix3+1) = (K+1)*(L-1:-1:2);
ix = cumsum([L;dix]);
[ir,ic] = ind2sub([K^2,2*L-1],ix);
T_KK = sparse(ir,ic,ones(numel(ir),1),K^2,2*L-1);

% This vector is used to replicate entries of vector phi of lenght L in
% vectorization of the KxK Toeplitz matrix,
% i.e. phi(phimap) = nonzero(Phi)

phimap = nonzeros(T_KK*[1:2*L-1]');
[rows0,cols0] = find(reshape(sum(T_KK,2),[K K]));

% For a Toeplitz matrix Phi of size KxK generated from vector phi of length L,
% Phi is a sparse matrix defined as
% Phi = sparse(rows0,cols0,phi(phimap),K,K);

krondim = [1:mode-1 mode+1:N];

LT = size(Y,1);
if nargout > 1
    W = zeros(SzU(mode,1)*R,LT*prod(SzU([1:mode-1 mode+1:end],2)));
end


% Estimate Hr
% Psi = zeros(L*R);
Vvec = zeros(SzY(mode+1)*L,R);

Hm = tenmat(H,mode+1);
Hm = reshape(Hm.data,SzU(mode,2), [], R);

rcv = zeros(R^2*nnzK,3);
cnt = 0;
for r = 1:R
    for s = r:R
        
        temp = U{krondim(1)}(:,:,r)'*U{krondim(1)}(:,:,s);
        for n = krondim(2:end)
            temp = kron(U{n}(:,:,r)'*U{n}(:,:,s),temp);
        end
        temp = Hm(:,:,r) * temp  * Hm(:,:,s)'; % ttt(ttensor(Hr,Ur),ttensor(Hs,Us),-mode)
        
        cnt = cnt+1;
        phi = T_LL*temp(:); % sums along diagonal of matrix temp
        
        vals = phi(phimap);
        rows = rows0+K*(r-1);
        cols = cols0+K*(s-1);
        rcv(nnzK*(cnt-1)+1:nnzK*cnt,:) = [rows cols vals];
        
        if s~=r
            cnt = cnt+1;
            rcv(nnzK*(cnt-1)+1:nnzK*cnt,:) = [cols rows vals];
        end
        
        %         Psi(L*(r-1)+1:L*r,L*(s-1)+1:L*s) = temp;
        %         Psi(L*(s-1)+1:L*s,L*(r-1)+1:L*r) = temp';
    end
    
    Ur = cellfun(@(x) x(:,:,r),U,'uni',0);
    %zy = ttm(Y,Ur(krondim),krondim+1,'t');
    zy = mttm(Y,Ur,mode+1);
    zy = tenmat(zy,mode+1);
    
    if nargout > 1
        % The following is for update core tensor H_r
        W(SzU(mode)*(r-1)+1:SzU(mode)*r,:) = zy; % Yx {U_r}
    end
    
    
    zy = zy * Hm(:,:,r)';
    Vvec(:,r) = zy(:);
end

%% Compute s
rcv(rcv(:,1) == 0,:) =[];
Phi = sparse(rcv(:,1),rcv(:,2),rcv(:,3),K*R,K*R);

w = T_KJ'*Vvec;w = w(:);
% un = MM\w;
% un = (MM+1e-12*speye(size(MM)))\w(:);

un = squeeze(U{mode}(1:SzU(mode)-L+1,1,:));
x = un(:);

% %% APG
%
% t0 = 1; t = t0;w = 0;xold = x;
% lambda = max(1,1/norm(full(MM)));lambdaold = lambda;
% % APG algorithm to estimate sparse un
% fx = (x(:)'*MM * x - 2 * w(:)'*x(:));
%
%
% for kiter1 = 1:5
%     y = x + w * (x-xold);
%     grad = -w + MM * y;
%
%     lambda_new = lambda;
%     for kiter2= 1:50
%         % gradient
%
%         param2.verbose=0;
%         param2.max_iter=10;
%
%         %z = max(0,abs(y - lambda_new * grad) - lambda_new) .* sign((y - lambda_new * grad));
%         z = prox_l1(y - lambda_new * grad, lambda_new, param2);
% %         z = max(0,y - lambda_new * grad);
%         fz = (z(:)'*MM * z- 2 * w(:)'*z(:));
%
%         if fz < fx
%             break
%         else
%             lambda_new = lambda_new/2;
%         end
%     end
%     xold = x; x = z; lambda = lambda_new;
%
%     t = (1+sqrt(1+4*t0^2))/2;
%     %get new extrapolated points
%     w = min([(t0-1)/t,sqrt(lambdaold/lambda)]);
%     lambdaold = lambda;
%
%     if abs(fx-fz)<1e-4*fx
%         break
%     end
%     fx = fz;
% end


%% ADMM % setting different parameter for the simulation
% param.verbose=0; % display parameter
% param.max_iter=300; % maximum iteration
% param.epsilon=10e-5; % tolerance to stop iterating
% param.gamma=1e-4; % stepsize
%
% tau = 1;
% % setting the function f2
% f2.prox=@(x,T) (T*MM + tau * eye(size(MM)))\(T*w + tau*x);
% f2.eval=@(x) ((x(:)'*MM * x(:) - 2 * w(:)'*x(:)))/2;
%
% % setting the function f1 (ell 1)
% param2.verbose=0;
% param2.max_iter=1;
%
% f1.prox=@(x, T) prox_l1(x, tau*T, param2);
% f1.eval=@(x) tau*sum(abs(x));
%
% fcur = tau*f1.eval(x) + f2.eval(x);
% for ki = 1:10
%     % Eventually, this choice is possible (Handle carefully)
%     % solving the problem
% %     [x2,infos]=admm(x,f2,f1, @(x) x,param);
%     [x2,infos]=admm(x,f2,f1, @(x) x,param);
%     % plot(x)
%     if infos.final_eval > fcur
%         param.gamma=param.gamma/2;
%     else
%         x = x2;
% %         if abs(infos.final_eval - fcur)<1e-3*fcur
% %             break
% %         end
%         fcur = infos.final_eval;
%         break
%     end
% end

param2.verbose=0;
tau = 0.01*max(abs(w(:)));lambda = 1 ; % max(1,1/norm(full(MM)));
u = 0; z = x;
% f1.prox=@(x,lambda) (lambda*MM + eye(size(MM)))\(lambda*w + x);

if size(Phi,1) < 500
    iPhi = inv(full(Phi + 1/lambda*eye(size(Phi))));
    temp = iPhi*w;
    f1.prox=@(x,lambda) temp + iPhi* x/lambda;
else
    f1.prox=@(x,lambda) (lambda*Phi + speye(size(Phi)))\(lambda*w + x);
end
objective = @(x,z) 1/2*(normY2 + x'*Phi*x - 2 * x'*w) + tau * sum(abs(z));
% ADMM-2
for k = 1:400
    x = f1.prox(z-u, lambda);
    
    % z-update
    %     zold = z;
    z = prox_l1(x+u, tau*lambda, param2);
    
    % u-update
    u = u + x - z;
    
    % diagnostics, reporting, termination checks
    h.admm_optval(k)   = objective(x, z);
    %     fprintf('iter %d Cost %d Rank(x) %d\n',k,h.admm_optval(k),rank(x))
    
    %     if norm(z-zold)< 1e-4*norm(zold)
    %         break
    %     end
    
    if  (k>1) && (abs(h.admm_optval(k) -  h.admm_optval(k-1))< h.admm_optval(k-1)*1e-3)
        break
    end
end
% x = z;


%% Update new Un
un = reshape(x,size(un));
un = reshape(un, [], R);
ell = sqrt(sum(un.^2));
H = reshape(bsxfun(@times,reshape(H,[],R),ell),size(H));

un = bsxfun(@rdivide,un,ell);
% un = [zeros(L-1,R); un; zeros(L-1,R)];
Un = zeros(SzU(mode,1),L,R);
for r = 1:R
    Un(:,:,r) = convmtx(un(:,r),L);
    %     Un(:,:,r) = toeplitz(un(L:SzU(mode,1)+L-1,r),un(L:-1:1,r)');
    %     Un(:,:,r) = mtoeplitz(un(L:SzU(mode,1),r),L);
end


if nargout > 1
    Vvec = zeros(SzU(mode,2)*R,LT*prod(SzU([1:mode-1 mode+1:end],2)));
    
    for r = 1:R
        Vvec(SzU(mode,2)*(r-1)+1:SzU(mode,2)*r,:) = Un(:,:,r)'*W(SzU(mode,1)*(r-1)+1:SzU(mode,1)*r,:);
    end
    
    Vvec = reshape(Vvec,[SzU(mode,2),R,SzU([1:mode-1 mode+1:end],2)',LT]);
    Vvec = permute(Vvec,[3:mode+1 1 mode+2:N+1 2 N+2]);
    Vvec = reshape(Vvec,[],LT);
end

end

%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [HNew,approx_error] = update_H(Y,U,Vvec)

SzU = cellfun(@(x) size(x),U,'uni',0);
SzU = cell2mat(SzU(:));
if size(SzU,2) == 2
    SzU(:,3) = 1; % R = 1;
end
R = SzU(1,end);
N = numel(U);

LN = prod(SzU(:,2)); LT = size(Y,1);

Vvecconstruct = false;
if (nargin<3) || isempty(Vvec)
    Vvecconstruct = true;
    Vvec = zeros(LN*R,LT);
end

% Construct the matrix Psi as in (16)
Psi = zeros(LN*R,LN*R);
for r = 1:R
    for s = r:R
        temp = U{1}(:,:,r)'*U{1}(:,:,s);
        for n = 2:N
            temp = kron(U{n}(:,:,r)'*U{n}(:,:,s),temp);
        end
        Psi(LN*(r-1)+1:LN*r,LN*(s-1)+1:LN*s) = temp;
        Psi(LN*(s-1)+1:LN*s,LN*(r-1)+1:LN*r) = temp';
    end
    
    if Vvecconstruct
        Ur = cellfun(@(x) x(:,:,r),U,'uni',0);
        zy = ttm(Y,Ur,2:N+1,'t');
        zy = reshape(double(zy),[],LT);
        
        Vvec(LN*(r-1)+1:LN*r,:) = zy;
    end
end

% Update H as in equation (15)
% HH = Psi\Vvec;
HH = (Psi+1e-12*speye(size(Psi)))\Vvec;
HNew = reshape(HH', [LT SzU(:,2)' R]);
% HNew = double(HNew);

% For fast computation of the Frobenious norm cost function
approx_error = -Vvec(:)'*HH(:);
end

%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [HNew,approx_error] = hupdate_H(Y,U,H,Vvec)

SzU = cellfun(@(x) size(x),U,'uni',0);
SzU = cell2mat(SzU(:));
R = SzU(1,end);
N = numel(U);

LN = prod(SzU(:,2)); LT = size(Y,1);
% Vvecconstruct = false;
% if (nargin<4) || isempty(Vvec)
%     Vvecconstruct = true;
%     Vvec = zeros(LN*R,LT);
% end

H = reshape(H,[],R);
for r = 1:R
    
    psi = 0;
    for s = [1:r-1 r+1:R]
        temp = U{1}(:,:,r)'*U{1}(:,:,s);
        for n = 2:N
            temp = kron(U{n}(:,:,r)'*U{n}(:,:,s),temp);
        end
        psi = psi + temp * H(:,s);
    end
    
    psi = Vvec(LN*(r-1)+1:LN*r,:) - psi;
    psi = reshape(psi,SzU(:,2)');
    for n = 1:N
        psi = reshape(psi,prod(SzU(1:n-1,2)), SzU(n,2),[]);
        psi = permute(psi,[2 1 3]);
        psi  = reshape(psi,SzU(n,2),[]);
        psi = (U{n}(:,:,r)'*U{n}(:,:,r) + 1e-12*eye(SzU(n,2)))\psi;
        psi = itenmat(psi,n,SzU(:,2)');
    end
    H(:,r) = psi(:);
end
HNew = reshape(H,[LT SzU(:,2)' R]);

% For fast computation of the Frobenious norm cost function
approx_error = -Vvec(:)'*H(:);
end


%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [HNew,approx_error] = update_Hsparse(Y,U,H,normY2,Vvec)

SzU = cellfun(@(x) size(x),U,'uni',0);
SzU = cell2mat(SzU(:));
R = SzU(1,end);
N = numel(U);

LN = prod(SzU(:,2)); LT = size(Y,1);
Psi = zeros(LN*R,LN*R);

Vvecconstruct = false;
if (nargin<5) || isempty(Vvec)
    Vvecconstruct = true;
    Vvec = zeros(LN*R,LT);
end


for r = 1:R
    for s = r:R
        temp = U{1}(:,:,r)'*U{1}(:,:,s);
        for n = 2:N
            temp = kron(U{n}(:,:,r)'*U{n}(:,:,s),temp);
        end
        Psi(LN*(r-1)+1:LN*r,LN*(s-1)+1:LN*s) = temp;
        Psi(LN*(s-1)+1:LN*s,LN*(r-1)+1:LN*r) = temp';
    end
    
    Ur = cellfun(@(x) x(:,:,r),U,'uni',0);
    if Vvecconstruct
        zy = ttm(Y,Ur,2:N+1,'t');
        zy = reshape(double(zy),[],LT);
        Vvec(LN*(r-1)+1:LN*r,:) = zy;
    end
end
% HH = (Psi+1e-12*speye(size(Psi)))\Vvec;
% HNew = reshape(HH', [LT SzU(:,2)' R]);
% HNew = double(HNew);

% approx_error = -Vvec(:)'*HH(:);

x = H(:);
% param2.verbose=0;

tau = 0.001*max(abs(Vvec(:)));lambda = 1 ; %max(1,1/norm(full(Psi)));
u = 0; z = x;
%f1.prox=@(x,lambda) (lambda*Psi + eye(size(Psi)))\(lambda*Vvec(:) + x);
iPhi = inv(lambda*Psi + eye(size(Psi)));
f1.prox=@(x,lambda) iPhi*(lambda*Vvec(:) + x);
feval = @(x,z) (normY2 + x'*Psi * x  - 2 * x'*Vvec(:))/2 + tau * sum(abs(z(:)));
% ADMM-2
fval = 0;
for k = 1:400
    x = f1.prox(z-u, lambda);
    
    % z-update
    %     zold = z;
    %z = prox_l1(x+u, tau*lambda, param2);
    z = max(0,x+u - tau*lambda) - max(0,-x-u - tau*lambda);
    
    % u-update
    u = u + x - z;
    
    
    fval(k) = feval(x,z)/normY2;
    
    if (k>1) && abs(fval(k-1)-fval(k))< 1e-4*fval(k-1)
        break
    end
end
x = z;

HNew = reshape(x,size(H));
approx_error = x'*Psi*x - 2 * x'*Vvec(:) + 2*tau * sum(abs(x));

end

  
%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

function X = mttm(Y,U,mode)
% This routine computes multplications Y x_{-mode}{U} when Lt i.e. size(Y,1) == 1.
%
RU = cellfun(@(x) size(x,2),U);
SzU = cellfun(@(x) size(x,1),U);
SzY = size(Y);
N = ndims(Y);
dim = 1:N;
if mode < N
    Y = reshape(Y,[],SzU(end)) * U{end};
    SzY(end) = RU(end);
    
    for kright = N-1:-1:mode+1
        right = prod(SzY(kright+1:end));
        if right ~= 1
            Y = reshape(Y,[],SzY(kright),right);
            Y = permute(Y,[1 3 2]);
            SzY(kright:end) = SzY([kright+1:end kright]);
            dim(kright:end) = dim([kright+1:end kright]);
        end
        Y =  reshape(Y,[],SzU(kright-1)) *U{kright-1};
        if right ~= 1
            SzY(end) = RU(kright-1);
        else
            SzY(kright) = RU(kright-1);
        end
    end
end

if mode > 2 % since the first dim of Y == 1
    Y =  U{1}' * reshape(Y,SzY(2),[]);
    SzY(2) = RU(1);
    
    for kleft = 3:mode-1
        left = prod(SzY(1:kleft-1));
        if left ~= 1
            Y = reshape(Y,left,SzY(kleft),[]);
            Y = permute(Y,[2 1 3]);
        end
        SzY(1:kleft) = SzY([kleft 1:kleft-1]);
        dim(1:kleft) = dim([kleft 1:kleft-1]);
        Y =  U{kleft-1}' * reshape(Y,SzU(kleft-1),[]);
        SzY(1) = RU(kleft-1);
    end
end

% ipermute
X = reshape(Y,SzY);
if ~issorted(dim)
    X = ipermute(X,dim);
end
end


%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

function x = block_levinson(y, L)
%BLOCK_LEVINSON Block Levinson recursion for efficiently solving
%symmetric block Toeplitz matrix equations.
%   BLOCK_LEVINSON(Y, L) solves the matrix equation T * x = y, where T
%   is a symmetric matrix with block Toeplitz structure, and returns the
%   solution vector x. The matrix T is never stored in full (because it
%   is large and mostly redundant), so the input parameter L is actually
%   the leftmost "block column" of T (the leftmost d columns where d is
%   the block dimension).

%   Author: Keenan Pepper
%   Last modified: 2007-12-23

%   References:
%     [1] Akaike, Hirotugu (1973). "Block Toeplitz Matrix Inversion".
%     SIAM J. Appl. Math. 24 (2): 234-241

s = size(L);
d = s(2);                 % Block dimension
N = s(1) / d;             % Number of blocks

B = reshape(L, [d,N,d]);  % This is just to get the bottom block row B
B = permute(B, [1,3,2]);  % from the left block column L
B = flipdim(B, 3);
B = reshape(B, [d,N*d]);

f = inv(L(1:d,:));          % "Forward" block vector
b = f;                    % "Backward" block vector
x = zeros(N*d,1);
x(1:d) = f * y(1:d);           % Solution vector

for n = 2:N
    % Since B and L comprise some zeros blocks of size d x d,
    % ef and eb can be computed faster.
    ef = B(:,(N-n)*d+1:(N-1)*d) * f; %ef = B(:,(N-n)*d+1:N*d) * [f;zeros(d)];
    eb = L(d+1:n*d,:)' * b; %eb = L(1:n*d,:)' * [zeros(d);b];
    ex = B(:,(N-n)*d+1:(N-1)*d) * x(1:d*(n-1));%ex = B(:,(N-n)*d+1:N*d) * [x;zeros(d,1)];
    %     A = [eye(d),eb;ef,eye(d)]^-1;
    %A1 = inv(eye(d) - eb*ef);A4 = inv(eye(d) - ef*eb);A2 = -eb*A4;A3 = -ef*A1;A = [A1 A2;A3 A4];
    %f = [[f;zeros(d)],[zeros(d);b]] * A(:,1:d);
    % b = [[f;zeros(d)],[zeros(d);b]] * A(:,d+1:end);
    fb = [[f;zeros(d)],[zeros(d);b]]/[eye(d),eb;ef,eye(d)];
    f = fb(:,1:d);b = fb(:,d+1:end);
    x(1:d*n) = x(1:d*n) + b * (y((n-1)*d+1:n*d) - ex);
end

end


%%
%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [Un,approx_error,Vvec] = Update_Uortho2(Y,U,H,mode,normY2)
% U:  cellarray of N factor matrices U{n} of size In x J1 x R, whose each
% slice U{n}(:,:,r) is Toeplitz matrix of the loading component u{n}_r (In)
SzH = size(H); % Lt x J1 x ...x JN x R
SzY = size(Y); % Lt x I1 x ...x IN
if ndims(H) == ndims(Y)
    SzH(end+1) = 1; % means that R == 1
end
SzU = [SzY(2:end)' SzH(2:end-1)'];
R = SzH(end);
L = SzU(mode,2);K = SzU(mode,1)-L+1;
Y = double(Y);
N = numel(U);%Y = tensor(Y);

% Toeplitz operator for a Toeplitz matrix of size (J*L) x (J+L-1)
% ri = reshape(1:L*SzU(mode,1),L,SzU(mode,1));ri =  ri(:);
% ci = bsxfun(@plus,(L:-1:1)',0:SzU(mode,1)-1)'; ci = ci(:);
% T_KJ = sparse(ri,ci,ones(L*SzU(mode,1),1),SzU(mode,1)*L,SzU(mode,1)+L-1);
% T_KJ = T_KJ(:,L:SzU(mode,1));

cols = repmat((1:K)',L,1);
rows = bsxfun(@plus,(1:K)',0:SzU(mode,1)+1:(SzU(mode,1)+1)*(L-1));
T_KJ = sparse(rows(:),cols(:),ones(numel(rows),1),SzU(mode,1)*L,K);

% Toeplitz operator to generate an LxL Toeplitz matrix from a vector of lenght (2L+1)
% vec(X) = T_LL * x
% Note that T_LL is used to sum diagonals of a matrix X of size LxL:  T_LL * vec(X)
ri = reshape(1:L*L,L,L);ri =  ri(:);
ci = bsxfun(@plus,(L:-1:1)',0:L-1)'; ci = ci(:);
T_LL = sparse(ri,ci,ones(L*L,1),L*L,L+L-1)';

% Toeplitz operator to generate a banded Toeplitz matrix of size K x K from vectors phi Lx1
nnzK = (2*L-1)*K - L*(L-1);
dix = (K+1)*ones(nnzK-1,1);
ix3 = cumsum(K-L+1:K-2);
dix(ix3) = (K+1)*(L-1:-1:2);
dix(end-ix3+1) = (K+1)*(L-1:-1:2);
ix = cumsum([L;dix]);
[ir,ic] = ind2sub([K^2,2*L-1],ix);
T_KK = sparse(ir,ic,ones(numel(ir),1),K^2,2*L-1);

% This vector is used to replicate entries of vector phi of lenght L in
% vectorization of the KxK Toeplitz matrix,
% i.e. phi(phimap) = nonzero(Phi)

phimap = nonzeros(T_KK*[1:2*L-1]');
[rows0,cols0] = find(reshape(sum(T_KK,2),[K K]));


krondim = [1:mode-1 mode+1:N];

% Prepare the common computation in estimation Un and core tensors H_r
LT = size(Y,1); % this parameter is often of 1, unless the data is converted to Toeplitz structure.
if nargout > 1
    W = zeros(SzU(mode,1)*R,LT*prod(SzU([1:mode-1 mode+1:end],2)));
end

% Estimate Hr
Vvec = zeros(SzY(mode+1)*L,R);

Hm = tenmat(H,mode+1);
Hm = reshape(Hm.data,SzU(mode,2), [], R);

rcv = ones(R^2*nnzK,3); % matrix of [rows cols values] of nonzero entries of the matrix Phi
cnt = 0;
for r = 1:R
    for s = r:R
        
        temp = U{krondim(1)}(:,:,r)'*U{krondim(1)}(:,:,s);
        for n = krondim(2:end)
            temp = kron(U{n}(:,:,r)'*U{n}(:,:,s),temp);
        end
        temp = Hm(:,:,r) * temp  * Hm(:,:,s)'; % ttt(ttensor(Hr,Ur),ttensor(Hs,Us),-mode)
        
        cnt = cnt+1;
        phi = T_LL*temp(:); % sums along diagonal of matrix temp
        
        %phi2 = T_KK*phi;
        %vals2 = nonzeros(phi2);
        vals = phi(phimap);
        
        rows = rows0+K*(r-1);
        cols = cols0+K*(s-1);
        rcv(nnzK*(cnt-1)+1:nnzK*cnt,:) = [rows cols vals];
        if s~=r
            cnt = cnt+1;
            rcv(nnzK*(cnt-1)+1:nnzK*cnt,:) = [cols rows vals];
        end
    end
    
    Ur = cellfun(@(x) x(:,:,r),U,'uni',0);
    %     zy = ttm(Y,Ur(krondim2),krondim2+1,'t');
    zy = mttm(Y,Ur,mode+1);
    zy = tenmat(zy,mode+1);
    
    if nargout > 2
        % The following is for update core tensor H_r
        W(SzU(mode)*(r-1)+1:SzU(mode)*r,:) = zy; % Yx {U_r}
    end
    
    zy = zy * Hm(:,:,r)';
    Vvec(:,r) = zy(:);
end

w = T_KJ'*Vvec;
w = reshape(w,[],R);
Phi = sparse(rcv(:,1),rcv(:,2),rcv(:,3),K*R,K*R);


%% CRNC
% Crank-Nicholson parameters
crnc_param = inputParser;
crnc_param.KeepUnmatched = true;
crnc_param.addParamValue('maxiters',10);
crnc_param.addParamValue('gtol',1e-5);
crnc_param.addParamValue('xtol',1e-5);
crnc_param.addParamValue('ftol',1e-8);
crnc_param.addParamValue('tau',1e-3);
crnc_param.addParamValue('rho',1e-4); % parameters for control the linear approximation in line search
crnc_param.addParamValue('eta',.2); % factor for decreasing the step size in the backtracking line search
crnc_param.addParamValue('gamma',.85); % updating C
crnc_param.addParamValue('nt',5);
crnc_param.addParamValue('iscomplex',0);
crnc_param.addParamValue('opt_stepsize',false);   % optimal step size or using the Barzilai-Borwein method
crnc_param.addParamValue('refine_stepsize',false);
crnc_param.addParamValue('stepsize', 'barzilai-borwein', @(x) (ismember(x,{'bb' 'barzilai-borwein' 'Barzilai-Borwein' 'exact'})));
% optimal step size or using the Barzilai-Borwein method

crnc_opts = struct;
crnc_param.parse(crnc_opts);
crnc_param = crnc_param.Results;


% crnc_param.projG = 2;
un = U{mode};
% un = [reshape(un(1,end:-1:2,:),[],R); squeeze(un(:,1,:))];
un = squeeze(un(1:SzU(mode,1)-L+1,1,:));

[un, out]= OptStiefelGBB(un, @deconv_cost, crnc_param,normY2,w,Phi);

%% Update new Un
un = reshape(un, [], R);
un = bsxfun(@rdivide,un,sqrt(sum(un.^2)));
un = [zeros(L-1,R); un; zeros(L-1,R)];
Un = zeros(SzU(mode,1),L,R);
for r = 1:R
    Un(:,:,r) = toeplitz(un(L:SzU(mode,1)+L-1,r),un(L:-1:1,r)');
    %     Un(:,:,r) = mtoeplitz(un(L:SzU(mode,1),r),L);
end

approx_error = sqrt(2*out.fval/normY2);


% Return the vector product Y x {Ur} for update Hn
if nargout > 2
    Vvec = zeros(SzU(mode,2)*R,LT*prod(SzU([1:mode-1 mode+1:end],2)));
    
    for r = 1:R
        Vvec(SzU(mode,2)*(r-1)+1:SzU(mode,2)*r,:) = Un(:,:,r)'*W(SzU(mode,1)*(r-1)+1:SzU(mode,1)*r,:);
    end
    
    Vvec = reshape(Vvec,[SzU(mode,2),R,SzU([1:mode-1 mode+1:end],2)',LT]);
    Vvec = permute(Vvec,[3:mode+1 1 mode+2:N+1 2 N+2]);
    Vvec = reshape(Vvec,[],LT);
end

end

%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [Un,approx_error] = Update_Uortho(Y,U,H,mode,normY2)
% U:  cellarray of N factor matrices U{n} of size In x J1 x R, whose each
% slice U{n}(:,:,r) is Toeplitz matrix of the loading component u{n}_r (In)
SzH = size(H); % Lt x J1 x ...x JN x R
SzY = size(Y); % Lt x I1 x ...x IN
SzU = [SzY(2:end)' SzH(2:end-1)'];
R = SzH(end);
L = SzU(mode,2);

N = numel(U);%Y = tensor(Y);

ri = reshape(1:L*SzU(mode,1),L,SzU(mode,1));ri =  ri(:);
ci = bsxfun(@plus,(L:-1:1)',0:SzU(mode,1)-1)'; ci = ci(:);
Tx = sparse(ri,ci,ones(L*SzU(mode,1),1),SzU(mode,1)*L,SzU(mode,1)+L-1);
Tx = Tx(:,L:SzU(mode,1));
Tx2 = kron(speye(R),Tx);

krondim = [1:mode-1 mode+1:N];


% Estimate Hr
ZZ2 = zeros(L*R);
ZY = zeros(SzY(mode+1)*L,R);

Hm = tenmat(H,mode+1);
Hm = reshape(Hm.data,SzU(mode,2), [], R);
for r = 1:R
    for s = r:R
        
        temp = U{krondim(1)}(:,:,r)'*U{krondim(1)}(:,:,s);
        for n = krondim(2:end)
            temp = kron(U{n}(:,:,r)'*U{n}(:,:,s),temp);
        end
        temp = Hm(:,:,r) * temp  * Hm(:,:,s)'; % ttt(ttensor(Hr,Ur),ttensor(Hs,Us),-mode)
        
        ZZ2(L*(r-1)+1:L*r,L*(s-1)+1:L*s) = temp;
        ZZ2(L*(s-1)+1:L*s,L*(r-1)+1:L*r) = temp';
    end
    
    Ur = cellfun(@(x) x(:,:,r),U,'uni',0);
    zy = ttm(Y,Ur(krondim),krondim+1,'t');
    zy = tenmat(zy,mode+1);
    zy = zy * Hm(:,:,r)';
    
    %ZY(SzY(mode+1)*L*(r-1)+1:SzY(mode+1)*L*r) = zy(:);
    ZY(:,r) = zy(:);
end

%% Compute s
% tic
MM = Tx2' *kron(ZZ2,speye(SzU(mode,1))) * Tx2;
% toc

%%
MYs = Tx'*ZY;
% un = (MM+1e-12*speye(size(MM)))\MYs(:);


%% CRNC
% Crank-Nicholson parameters
crnc_param = inputParser;
crnc_param.KeepUnmatched = true;
crnc_param.addParamValue('maxiters',10);
crnc_param.addParamValue('gtol',1e-5);
crnc_param.addParamValue('xtol',1e-5);
crnc_param.addParamValue('ftol',1e-8);
crnc_param.addParamValue('tau',1e-3);
crnc_param.addParamValue('rho',1e-4); % parameters for control the linear approximation in line search
crnc_param.addParamValue('eta',.2); % factor for decreasing the step size in the backtracking line search
crnc_param.addParamValue('gamma',.85); % updating C
crnc_param.addParamValue('nt',5);
crnc_param.addParamValue('iscomplex',0);
crnc_param.addParamValue('opt_stepsize',false);   % optimal step size or using the Barzilai-Borwein method
crnc_param.addParamValue('refine_stepsize',false);
crnc_param.addParamValue('stepsize', 'barzilai-borwein', @(x) (ismember(x,{'bb' 'barzilai-borwein' 'Barzilai-Borwein' 'exact'})));
% optimal step size or using the Barzilai-Borwein method

crnc_opts = struct;
crnc_param.parse(crnc_opts);
crnc_param = crnc_param.Results;


% crnc_param.projG = 2;
un = U{mode};
% un = [reshape(un(1,end:-1:2,:),[],R); squeeze(un(:,1,:))];
un = squeeze(un(1:SzU(mode,1)-L+1,1,:));

[un, out]= OptStiefelGBB(un, @deconv_cost, crnc_param,normY2,MYs,MM);

%% Update new Un
un = reshape(un, [], R);
un = bsxfun(@rdivide,un,sqrt(sum(un.^2)));
un = [zeros(L-1,R); un; zeros(L-1,R)];
Un = zeros(SzU(mode,1),L,R);
for r = 1:R
    Un(:,:,r) = toeplitz(un(L:SzU(mode,1)+L-1,r),un(L:-1:1,r)');
    %     Un(:,:,r) = mtoeplitz(un(L:SzU(mode,1),r),L);
end

approx_error = sqrt(2*out.fval/normY2);
end



function [f,g] = deconv_cost(u,normY2,MYs,MM)
% cost function of deconvolution for CrNc update
% f: cost value
% g: gradient of f w.r.t to u

% [I,L,R] = size(U{mode});
% un = reshape(u, [], R);
% Un = zeros([I,L,R]);
% for r = 1:R
%     Un(:,:,r) = toeplitz(un(L:I+L-1,r),un(L:-1:1,r)');
% end

% U{mode} = Un;
% f = normFrob(Y,U,H,normY2)^2/2;
temp = MM * u(:);
f = (normY2 + u(:)'*temp - 2 * MYs(:)'*u(:))/2;
% gradient

g = -MYs + reshape(temp,size(u));

end



function [X, out]= OptStiefelGBB(X, fun, opts, varargin)
%-------------------------------------------------------------------------
% curvilinear search algorithm for optimization on Stiefel manifold
%
%   min F(X), S.t., X'*X = I_k, where X \in R^{n,k}
%
%   H = [G, X]*[X -G]'
%   U = 0.5*tau*[G, X];    V = [X -G]
%   X(tau) = X - 2*U * inv( I + V'*U ) * V'*X
%
%   -------------------------------------
%   U = -[G,X];  V = [X -G];  VU = V'*U;
%   X(tau) = X - tau*U * inv( I + 0.5*tau*VU ) * V'*X
%
%
% Input:
%           X --- n by k matrix such that X'*X = I
%         fun --- objective function and its gradient:
%                 [F, G] = fun(X,  data1, data2)
%                 F, G are the objective function value and gradient, repectively
%                 data1, data2 are addtional data, and can be more
%                 Calling syntax:
%                   [X, out]= OptStiefelGBB(X0, @fun, opts, data1, data2);
%
%        opts --- option structure with fields:
%                 record = 0, no print out
%                 mxitr       max number of iterations
%                 xtol        stop control for ||X_k - X_{k-1}||
%                 gtol        stop control for the projected gradient
%                 ftol        stop control for |F_k - F_{k-1}|/(1+|F_{k-1}|)
%                             usually, max{xtol, gtol} > ftol
%
% Output:
%           X --- solution
%         Out --- output information
%
% -------------------------------------
% For example, consider the eigenvalue problem F(X) = -0.5*Tr(X'*A*X);
%
% function demo
%
% function [F, G] = fun(X,  A)
%   G = -(A*X);
%   F = 0.5*sum(dot(G,X,1));
% end
%
% n = 1000; k = 6;
% A = randn(n); A = A'*A;
% opts.record = 0; %
% opts.mxitr  = 1000;
% opts.xtol = 1e-5;
% opts.gtol = 1e-5;
% opts.ftol = 1e-8;
%
% X0 = randn(n,k);    X0 = orth(X0);
% tic; [X, out]= OptStiefelGBB(X0, @fun, opts, A); tsolve = toc;
% out.fval = -2*out.fval; % convert the function value to the sum of eigenvalues
% fprintf('\nOptM: obj: %7.6e, itr: %d, nfe: %d, cpu: %f, norm(XT*X-I): %3.2e \n', ...
%             out.fval, out.itr, out.nfe, tsolve, norm(X'*X - eye(k), 'fro') );
%
% end
% -------------------------------------
%
% Reference:
%  Z. Wen and W. Yin
%  A feasible method for optimization with orthogonality constraints
%
% Author: Zaiwen Wen, Wotao Yin
%   Version 1.0 .... 2010/10
%-------------------------------------------------------------------------


%% Size information
if isempty(X)
    error('input X is an empty matrix');
else
    [n, k] = size(X);
end

if isfield(opts, 'xtol')
    if opts.xtol < 0 || opts.xtol > 1
        opts.xtol = 1e-6;
    end
else
    opts.xtol = 1e-6;
end

if isfield(opts, 'gtol')
    if opts.gtol < 0 || opts.gtol > 1
        opts.gtol = 1e-6;
    end
else
    opts.gtol = 1e-6;
end

if isfield(opts, 'ftol')
    if opts.ftol < 0 || opts.ftol > 1
        opts.ftol = 1e-12;
    end
else
    opts.ftol = 1e-12;
end

% parameters for control the linear approximation in line search
if isfield(opts, 'rho')
    if opts.rho < 0 || opts.rho > 1
        opts.rho = 1e-4;
    end
else
    opts.rho = 1e-4;
end

% factor for decreasing the step size in the backtracking line search
if isfield(opts, 'eta')
    if opts.eta < 0 || opts.eta > 1
        opts.eta = 0.1;
    end
else
    opts.eta = 0.2;
end

% parameters for updating C by HongChao, Zhang
if isfield(opts, 'gamma')
    if opts.gamma < 0 || opts.gamma > 1
        opts.gamma = 0.85;
    end
else
    opts.gamma = 0.85;
end

if isfield(opts, 'tau')
    if opts.tau < 0 || opts.tau > 1e3
        opts.tau = 1e-3;
    end
else
    opts.tau = 1e-3;
end

% parameters for the  nonmontone line search by Raydan
if ~isfield(opts, 'STPEPS')
    opts.STPEPS = 1e-10;
end

if isfield(opts, 'nt')
    if opts.nt < 0 || opts.nt > 100
        opts.nt = 5;
    end
else
    opts.nt = 5;
end

if isfield(opts, 'projG')
    switch opts.projG
        case {1,2}; otherwise; opts.projG = 1;
    end
else
    opts.projG = 1;
end

if isfield(opts, 'iscomplex')
    switch opts.iscomplex
        case {0, 1}; otherwise; opts.iscomplex = 0;
    end
else
    opts.iscomplex = 0;
end

if isfield(opts, 'mxitr')
    if opts.mxitr < 0 || opts.mxitr > 2^20
        opts.mxitr = 1000;
    end
else
    opts.mxitr = 1000;
end

if ~isfield(opts, 'record')
    opts.record = 0;
end


%-------------------------------------------------------------------------------
% copy parameters
xtol = opts.xtol;
gtol = opts.gtol;
ftol = opts.ftol;
rho  = opts.rho;
STPEPS = opts.STPEPS;
eta   = opts.eta;
gamma = opts.gamma;
iscomplex = opts.iscomplex;
record = opts.record;

nt = opts.nt;   crit = ones(nt, 3);

invH = true; if k < n/2; invH = false;  eye2k = eye(2*k); end

%% Initial function value and gradient
% prepare for iterations
[F,  G] = feval(fun, X , varargin{:});  out.nfe = 1;
GX = G'*X;

if invH
    GXT = G*X';  H = 0.5*(GXT - GXT');  RX = H*X;
else
    if opts.projG == 1
        U =  [G, X];    V = [X, -G];       VU = V'*U;
    elseif opts.projG == 2
        GB = G - 0.5*X*(X'*G);
        U =  [GB, X];    V = [X, -GB];       VU = V'*U;
    end
    %U =  [G, X];    VU = [GX', X'*X; -(G'*G), -GX];
    %VX = VU(:,k+1:end); %VX = V'*X;
    VX = V'*X;
end
dtX = G - X*GX;     nrmG  = norm(dtX, 'fro');

Q = 1; Cval = F;  tau = opts.tau;

%% Print out the results if debug == 1
if (opts.record == 1)
    fid = 1;
    fprintf(fid, '----------- Gradient Method with Line search ----------- \n');
    fprintf(fid, '%4s %8s %8s %10s %10s\n', 'Iter', 'tau', 'F(X)', 'nrmG', 'XDiff');
    %fprintf(fid, '%4d \t %3.2e \t %3.2e \t %5d \t %5d	\t %6d	\n', 0, 0, F, 0, 0, 0);
end

%% main iteration
for itr = 1 : opts.mxitr
    XP = X;     FP = F;   GP = G;   dtXP = dtX;
    % scale step size
    
    nls = 1; deriv = rho*nrmG^2; %deriv
    while 1
        % calculate G, F,
        if invH
            [X, infX] = linsolve(eye(n) + tau*H, XP - tau*RX);
        else
            [aa, infR] = linsolve(eye2k + (0.5*tau)*VU, VX);
            X = XP - U*(tau*aa);
        end
        %if norm(X'*X - eye(k),'fro') > 1e-6; error('X^T*X~=I'); end
        if ~isreal(X) && ~iscomplex ; error('X is complex'); end
        
        [F,G] = feval(fun, X, varargin{:});
        out.nfe = out.nfe + 1;
        
        if F <= Cval - tau*deriv || nls >= 5
            break;
        end
        tau = eta*tau;          nls = nls+1;
    end
    
    GX = G'*X;
    if invH
        GXT = G*X';  H = 0.5*(GXT - GXT');  RX = H*X;
    else
        if opts.projG == 1
            U =  [G, X];    V = [X, -G];       VU = V'*U;
        elseif opts.projG == 2
            GB = G - 0.5*X*(X'*G);
            U =  [GB, X];    V = [X, -GB];     VU = V'*U;
        end
        %U =  [G, X];    VU = [GX', X'*X; -(G'*G), -GX];
        %VX = VU(:,k+1:end); % VX = V'*X;
        VX = V'*X;
    end
    dtX = G - X*GX;    nrmG  = norm(dtX, 'fro');
    
    S = X - XP;         XDiff = norm(S,'fro')/sqrt(n);
    tau = opts.tau; FDiff = abs(FP-F)/(abs(FP)+1);
    
    if iscomplex
        %Y = dtX - dtXP;     SY = (sum(sum(real(conj(S).*Y))));
        Y = dtX - dtXP;     SY = abs(sum(sum(conj(S).*Y)));
        if mod(itr,2)==0; tau = sum(sum(conj(S).*S))/SY;
        else tau = SY/sum(sum(conj(Y).*Y)); end
    else
        %Y = G - GP;     SY = abs(sum(sum(S.*Y)));
        Y = dtX - dtXP;     SY = abs(sum(sum(S.*Y)));
        %alpha = sum(sum(S.*S))/SY;
        %alpha = SY/sum(sum(Y.*Y));
        %alpha = max([sum(sum(S.*S))/SY, SY/sum(sum(Y.*Y))]);
        if mod(itr,2)==0; tau = (sum(sum(S.*S))+eps)/(SY+eps);
        else tau  = (SY+eps)/(sum(sum(Y.*Y))+eps); end
        
        % %Y = G - GP;
        % Y = dtX - dtXP;
        % YX = Y'*X;     SX = S'*X;
        % SY =  abs(sum(sum(S.*Y)) - 0.5*sum(sum(YX.*SX)) );
        % if mod(itr,2)==0;
        %     tau = SY/(sum(sum(S.*S))- 0.5*sum(sum(SX.*SX)));
        % else
        %     tau = (sum(sum(Y.*Y)) -0.5*sum(sum(YX.*YX)))/SY;
        % end
        
    end
    tau = max(min(tau, 1e20), 1e-20);
    
    if (record >= 1)
        fprintf('%4d  %3.2e  %4.3e  %3.2e  %3.2e  %3.2e  %2d\n', ...
            itr, tau, F, nrmG, XDiff, FDiff, nls);
        %fprintf('%4d  %3.2e  %4.3e  %3.2e  %3.2e (%3.2e, %3.2e)\n', ...
        %    itr, tau, F, nrmG, XDiff, alpha1, alpha2);
    end
    
    crit(itr,:) = [nrmG, XDiff, FDiff];
    mcrit = mean(crit(itr-min(nt,itr)+1:itr, :),1);
    %if (XDiff < xtol && nrmG < gtol ) || FDiff < ftol
    %if (XDiff < xtol || nrmG < gtol ) || FDiff < ftol
    %if ( XDiff < xtol && FDiff < ftol ) || nrmG < gtol
    %if ( XDiff < xtol || FDiff < ftol ) || nrmG < gtol
    if ( XDiff < xtol && FDiff < ftol ) || nrmG < gtol || all(mcrit(2:3) < 10*[xtol, ftol])
        if itr <= 2
            ftol = 0.1*ftol;
            xtol = 0.1*xtol;
            gtol = 0.1*gtol;
        else
            out.msg = 'converge';
            break;
        end
    end
    
    Qp = Q; Q = gamma*Qp + 1; Cval = (gamma*Qp*Cval + F)/Q;
end

if itr >= opts.mxitr
    out.msg = 'exceed max iteration';
end

out.feasi = norm(X'*X-eye(k),'fro');
if  out.feasi > 1e-13
    X = MGramSchmidt(X);
    [F,G] = feval(fun, X, varargin{:});
    out.nfe = out.nfe + 1;
    out.feasi = norm(X'*X-eye(k),'fro');
end

out.nrmG = nrmG;
out.fval = F;
out.itr = itr;




end



function V = MGramSchmidt(V)
[n,k] = size(V);

for dj = 1:k
    for di = 1:dj-1
        V(:,dj) = V(:,dj) - proj(V(:,di), V(:,dj));
    end
    V(:,dj) = V(:,dj)/norm(V(:,dj));
end
end


%project v onto u
function v = proj(u,v)
v = (dot(v,u)/dot(u,u))*u;
end
