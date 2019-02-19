function U = seqproj_trunc(X,R)
% Sequential projection and truncation for best rank R
% 
% Like TT-SVD
% Phan Anh Huy, 2017
%
Nf = ndims(X);
U = cell(Nf,1);
dimorder = 1:Nf;
if numel(R)==1
    R = R(ones(1,Nf));
end
for n = 1:Nf-1
%     try
        if n==1
            U{dimorder(n)} = nvecs(X,dimorder(n),R(n));
        else
            if n==2
                T = ttm(X,U{dimorder(n-1)},n-1,'t');
            else
                T = ttm(T,U{dimorder(n-1)},n-1,'t');
            end
            U{dimorder(n)} = nvecs(T,dimorder(n),R((n)));
        end
%     catch me
%         Xn = double(tenmat(X,dimorder(n)));
%         Xn = Xn*Xn';
%         R((n)) = min(rank(Xn),R((n)));
%         U{dimorder(n)} = nvecs(X,dimorder(n),R((nwhich )));
%     end
end
U{dimorder(end)} = squeeze(double(ttm(T,U{dimorder(end-1)},Nf-1,'t')));