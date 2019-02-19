function decision = comparefoldings(fold1,fold2)
% Compare two two tensor unfolding rules fold1, fold2.
%
% decision = false : if they are different.
%
% This algorithm is a part of the TENSORBOX, 2012.
%
% Copyright Phan Anh Huy, 04/2012

decision = true;
if numel(fold1) ~= numel(fold2)
    decision = false; % different folding rules
    return
end

fold1 = cellfun(@sort,fold1,'uni',0);
fold2 = cellfun(@sort,fold2,'uni',0);

for k = 1:numel(fold1)
    fidx = cellfun(@(x) isequal(x,fold1{k}),fold2);
    if all(fidx==0)
        decision = false; % different folding rules
        return
    end
end
