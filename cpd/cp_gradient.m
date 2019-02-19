function [G,Pmat] = cp_gradient(Pmat,A,n)
%% mode-n CP Gradient of X with respect to A
% input X needs only as n == ns and n == ns+1, i.e.,
% if (n == ns) || (n == ns+1)
%     [Gxn{n},Pmat] = cp_gradient(X,U,n);   % compute CP gradients
% else
%     [Gxn{n},Pmat] = cp_gradient(Pmat,U,n);   % compute CP gradients
% end
% 
% Phan Anh Huy,
% TENSORBOX, 2012.
N = numel(A);
In = cellfun(@(x) size(x,1), A);
R = size(A{1},2);

if ~issorted(In)
    error('Rearrange data''s dimensions in ascending order : %s.\nThen try again.', ...
        [sprintf('%d x ',sort(In(1:end-1))) num2str(In(end))])
end

% Find the first n* such that I1...In* > I(n*+1) ... IN
Jn = cumprod(In); Kn = Jn(end)./Jn;
ns = find(Jn>Kn,1);
if ((ns >= (N-1)) && (ns > 2))
    ns = ns -1;
end

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


if n <= ns
    if n == ns
        if numel(right) == 1
            KRP_right = A{right};
        elseif numel(right) > 1
            KRP_right = khatrirao(A(right)); %KRP_right = khatrirao(A(right));
        end
        
        Pmat = reshape(double(Pmat),[],prod(In(right))); % Right-side projection
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
        T = reshape(Pmat,prod(In(left)),In(n),[]);
        if R>1
            T = bsxfun(@times,T,reshape(KRP_left,[],1,R));
            T = sum(T,1);
            G = squeeze(T);
        else
            G = (KRP_left'*T)';
        end
    else
        G = squeeze(Pmat);
    end
    
elseif n >=ns+1
    if n ==ns+1
        if numel(left) == 1
            KRP_left = A{left};
        elseif numel(left) > 1
            KRP_left = khatrirao(A(left));
        end
        T = reshape(double(Pmat),prod(In(left)),[]);
        Pmat = KRP_left' * T;   % Left-side projection
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
        KRP_right = khatrirao(A(right));
        if R>1
            T = bsxfun(@times,T,reshape(KRP_right',R,1,[]));
            T = sum(T,3);
            G = squeeze(T)';        % Right-side projection
        else
            G = squeeze(T) * KRP_right;
        end
    else
        G = squeeze(Pmat)';
    end
    
end
end
