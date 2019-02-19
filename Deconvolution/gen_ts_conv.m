function varargout = gen_ts_conv(varargin)
%    [X,H,U] = gen_ts_conv(SzU,SzH,R)
% or
%    X = gen_ts_conv(H,U)
%
% Generate a tensor X from core tensors (patterns) H_r and rank-1
% activating tensors whose loading components are U{n}(:,r).

% [X,H,U] = gen_ts_conv(SzU,SzH,R)
%
% TENSORBOX, 2018

if (nargin == 3) && isvector(varargin{1}) && isvector(varargin{2}) && isscalar(varargin{3})
    [SzU,SzH,R] = deal(varargin{1},varargin{2},varargin{3});
    
    N = numel(SzU);
    H = cell(N,1); U = cell(1,R);
    
    density = 100;
    
    X = 0;
    for r = 1:R
        Hr = randn(SzH);
        ur = cell(N,1);
        ortho = false;
        for n = 1:N
            ur{n} = randn(SzU(n),1);
            W = nmf_gen(SzU(n),1,density,ortho);
            ur{n} = ur{n}.*W;
        end
        
        H{r} = Hr;U{r} = ur;
    
 
        X = X + full(convn(Hr,ktensor(ur)));
    end
    
    %% Generate X from H and U
    %X = gen_ts_conv(H,U);
    
    % Set output
    if nargout >=1
        varargout{1} = X;
    end
    if nargout >=2
        varargout{2} = H;
    end
    if nargout >=3
        varargout{3} = U;
    end
    
elseif (nargin == 2)     %  X = gen_ts_conv(H,U)
    H = varargin{1};
    U = varargin{2};
    
    if ~iscell(H) % Block model of the convolutive tensor mixtures
        % H is an array of size "? x SzH(1) x SzH(2) x...xSzH(N) x R"
        % U: is a cell array, each entry comprises concatenation of
        % loadings of the same order, i.e., U{n} is of size (In+Jn-1) x Jn x R.
        szs = cell2mat(cellfun(@(x) size(x),U,'uni',0));
        SzX = szs(:,1);
        %SzH = szs(:,2);
        R = szs(1,3);
        N = numel(U);
        SzH = size(H);
        % H must be of size 
        Hr = reshape(double(H),[],R);
        X = 0;
        for r = 1:R
            X = X+ ttm(tensor(Hr(:,r),SzH(1:end-1)),...
                cellfun(@(x) x(:,:,r), U,'uni',0),2:N+1);
        end
        
        
    else % generate the tensor X from the convolutive model of H x U
        
        R = numel(H);
        X = 0;
        for r = 1:R
            % Generate the r-th term X_r, X = sum_r X_r.
            Xr = convn(H{r},ktensor(U{r}));
            X = X + Xr;
        end
    end
    varargout{1} = X;
end

%  H = reshape(cell2mat(cellfun(@(x) x(:),H(:)','uni',0)),[SzH,R]);
% 
% for n = 1:N
%     U{n,1} = cell2mat(U(n,:));
% end
% U = U(:,1);

% for n = 1:N
%     U{n,1} = cell2mat(cellfun(@(x) convmtx(x,SzH(n)),U(n,:),'uni',0));
%     U{n,1} = reshape(U{n,1},[SzU(n)+SzH(n)-1 SzH(n) R]);
% end
% U = U(:,1);
end



function Az = nmf_gen(I,R,density,orthogonal)

if nargin < 4
    orthogonal = 0;
end
% Sparse ratio
density = min(density,1-(R-1)/I(1));

if orthogonal
    % Generate orthogonal nonnegative matrix
    % Generate num of non-zeros for columns nnz_A = n1 + n2 + ... + nR
    nnz_A = 2/I + rand(1,R) * (1-(R-3)/I);
    nnz_A = I*nnz_A/sum(nnz_A);
    nnz_A = ceil(sort(nnz_A));
    nnz_A(end-(sum(nnz_A) - I)+1:end) = nnz_A(end-(sum(nnz_A) - I)+1:end)-1;
    nnz_A = nnz_A(randperm(R));
    
    innz = 1:I;
    innz = innz(randperm(I));
    cnnz_A = cumsum([0 nnz_A]);
    Az = zeros(I,R);
    for r = 1:R
        Az(innz(cnnz_A(r)+1:cnnz_A(r+1)),r) = 1;
    end
else
    while 1
        Az = binornd(1,density,[I(1) R]);
        dd = pdist(Az',@(x,y) sum(bsxfun(@and,x,y),2)>= bsxfun(@min,sum(y,2),sum(x)));
        if all(dd==0)
            break
        end
    end 
end

end