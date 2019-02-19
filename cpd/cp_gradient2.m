%% CP Gradient with respect to mode n
function [G,Pmat] = cp_gradient(A,n,Pmat)
persistent KRP_right0;

N = numel(A);
In = cellfun(@(x) size(x,1), A);In = In(:)';
R = size(A{1},2);

if ~issorted(In)
    error('Rearrange data''s dimensions in ascending order : %s.\nThen try again.', ...
        [sprintf('%d x ',sort(In(1:end-1))) num2str(In(end))])
end

% Find the first n* such that I1...In* > I(n*+1) ... IN
Jn = cumprod(In); Kn = [Jn(end) Jn(end)./Jn(1:end-1)];
ns = find(Jn<=Kn,1,'last');

right = N:-1:n+1; left = n-1:-1:1;

% check dimensions
if (n == ns) || (n == ns+1)
    if any(size(Pmat) ~= In(:)')
        error('Dimensions of data tensor and factors do not match.')
    end
elseif n< ns
    if any(numel(Pmat) ~= R*In(right(end))*In(n))
        warning('Matlab:invalid_dim',...
            'Invalid dimension. %s should have %d elements.\n',...
            inputname(1),R*prod(In(right(end)))*In(n))
    end
end

% KRP_right =[]; KRP_left = [];
if n <= ns
    if n == ns
        if numel(right) == 1
            KRP_right = A{right};
        elseif numel(right) > 2
            [KRP_right,KRP_right0] = khatrirao(A(right));
        elseif numel(right) > 1
            KRP_right = khatrirao(A(right));
        else
            KRP_right = 1;
        end
        
        if isa(Pmat,'tensor')
            Pmat = reshape(Pmat.data,[],prod(In(right))); % Right-side projection
        elseif isa(Pmat,'sptensor')
            Pmat = reshape(Pmat,[prod(size(Pmat))/prod(In(right)),prod(In(right))]); % Right-side projection
            Pmat = spmatrix(Pmat);
        else
            Pmat = reshape(Pmat,[],prod(In(right))); % Right-side projection
        end
        Pmat = Pmat * KRP_right ;
    else
        Pmat = reshape(Pmat,[],In(right(end)),R);
        if R>1
            Pmat = bsxfun(@times,Pmat,reshape(A{right(end)},[],In(right(end)),R));
            Pmat = sum(Pmat,2);    % fast Right-side projection
        else
            Pmat = Pmat * A{right(end)};
        end
    end
    
    if ~isempty(left)       % Left-side projection
        KRP_left = khatrirao(A(left));
        %                 if (isempty(KRP_2) && (numel(left) > 2))
        %                     [KRP_left,KRP_2] = khatrirao(A(left));
        %                 elseif isempty(KRP_2)
        %                     KRP_left = khatrirao(A(left));
        %                     %KRP_2 = [];
        %                 else
        %                     KRP_left = KRP_2; KRP_2 = [];
        %                 end
        T = reshape(Pmat,prod(In(left)),In(n),[]);
        if R>1
            T = bsxfun(@times,T,reshape(KRP_left,[],1,R));
            T = sum(T,1);
            %G = squeeze(T);
            G = reshape(T,[],R);
        else
            G = (KRP_left'*T)';
        end
    else
        %G = squeeze(Pmat);
        G = reshape(Pmat,[],R);
    end
    
elseif n >=ns+1
    if n ==ns+1
        if numel(left) == 1
            KRP_left = A{left}';
        elseif numel(left) > 1
            KRP_left = khatrirao_t(A(left));
            %KRP_left = khatrirao(A(left));KRP_left = KRP_left';
        else
            KRP_left = 1;
        end
        if isa(Pmat,'tensor')
            T = reshape(Pmat.data,prod(In(left)),[]);
        elseif isa(Pmat,'sptensor')
            T = reshape(Pmat,[prod(In(left)) prod(size(Pmat))/prod(In(left))]); % Right-side projection
            T = spmatrix(T);
        else
            T = reshape(Pmat,prod(In(left)),[]);
        end
        %
        Pmat = KRP_left * T;   % Left-side projection
    else
        if R>1
            Pmat = reshape(Pmat,R,In(left(1)),[]);
            Pmat = bsxfun(@times,Pmat,A{left(1)}');
            Pmat = sum(Pmat,2);      % Fast Left-side projection
        else
            Pmat = reshape(Pmat,In(left(1)),[]);
            Pmat = A{left(1)}'* Pmat;
        end
    end
    
    if ~isempty(right)
        T = reshape(Pmat,[],In(n),prod(In(right)));
        
        if (n == (ns+1)) && (numel(right)>=2)
            %KRP_right = KRP_right0;
            if R>1
                T = bsxfun(@times,T,reshape(KRP_right0',R,1,[]));
                T = sum(T,3);
                %G = squeeze(T)';        % Right-side projection
                G = reshape(T, R,[])';
            else
                %G = squeeze(T) * KRP_right0;
                G = reshape(T,[],prod(In(right))) * KRP_right0;
            end
        else
            KRP_right = khatrirao(A(right));
            if R>1
                T = bsxfun(@times,T,reshape(KRP_right',R,1,[]));
                T = sum(T,3);
                %G = squeeze(T)';        % Right-side projection
                G = reshape(T,R,[])';        % Right-side projection
            else
                %G = squeeze(T) * KRP_right;
                G = reshape(T,In(n),[]) * KRP_right;
            end
        end
    else
        %G = squeeze(Pmat)';
        G = reshape(Pmat,R,[])';
    end
    
end

%         fprintf('n = %d, Pmat %d x %d, \t Left %d x %d,\t Right %d x %d\n',...
%             n, size(squeeze(Pmat),1),size(squeeze(Pmat),2),...
%             size(KRP_left,1),size(KRP_left,2),...
%             size(KRP_right,1),size(KRP_right,2))
end


%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [K,K2] = khatrirao(A)
% Fast Khatri-Rao product
% P = fastkhatrirao({A,B,C})
% 
%Copyright 2012, Phan Anh Huy.

R = size(A{1},2);
K = A{1};
if nargout == 1
    for i = 2:numel(A)
        K = bsxfun(@times,reshape(A{i},[],1,R),reshape(K,1,[],R));
    end
elseif numel(A) > 2
    for i = 2:numel(A)-1
        K = bsxfun(@times,reshape(A{i},[],1,R),reshape(K,1,[],R));
    end
    K2 = reshape(K,[],R);
    K = bsxfun(@times,reshape(A{end},[],1,R),reshape(K,1,[],R));
end
K = reshape(K,[],R);
end


%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [K,K2] = khatrirao_t(A)
% Fast Khatri-Rao product
% P = fastkhatrirao({A,B,C})
% 
%Copyright 2012, Phan Anh Huy.

R = size(A{1},2);
K = A{1}';

for i = 2:numel(A)
    K = bsxfun(@times,reshape(A{i}',R,[]),reshape(K,R,1,[]));
end
K = reshape(K,R,[]);

end

%%
function K = kron(A,B)
%  Fast implementation of Kronecker product of A and B
%
%   Copyright 2012 Phan Anh Huy
%   $Date: 2012/3/18$

if ndims(A) > 2 || ndims(B) > 2
    error(message('See ndkron.m'));
end
I = size(A); J = size(B);

if ~issparse(A) && ~issparse(B)
    K = bsxfun(@times,reshape(B,J(1),1,J(2),1),reshape(A,1,I(1),1,I(2)));
    K = reshape(K,I(1)*J(1),[]);
else
    K = kron(A,B);
end
end