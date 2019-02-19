function X = bdiagtimematrix(A,B)
% A or B is a diagonal matrix
% TENSORBOX, 2018

if isstruct(A) && (strcmp(A.type,'bdiag'))
    if isstruct(B)
        for n = 1:size(A.mat,3)
            X(:,:,n) = A.mat(:,:,n) * B.mat(:,:,n);
        end
        X.type = B.type;
    else
        for n = 1:size(A.mat,3)
            X(:,:,n) = A.mat(:,:,n) * B();
        end
    end
else
end