function [H,U] = convolutive_to_bcd(H,U)
% Conversion of a convolutive model to a block tensor component
%
%
%  H is a cell array of Hr which should be of the same size.
%  U is a cell array of Ur
%
% TENSORBOX, 2018

SzH = size(H{1});
R = numel(H);
N = numel(U{1});

H = reshape(cell2mat(cellfun(@(x) x(:),H(:)','uni',0)),[1 SzH,R]);

U2 = cell(N,1);
for r = 1:R
    for n = 1:N
        if r == 1
            U2{n} = convmtx(U{r}{n},SzH(n));
        else
            U2{n} = cat(3,U2{n},convmtx(U{r}{n},SzH(n)));
        end
    end
end
U = U2;