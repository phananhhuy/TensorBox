function Y = cp_to_tt(A)
%
% Exact CP to TT conversion 
%
% Inputs are factor matrices A{n} 
% Output is a TT-tensor Y whose core tensors [[An,I,I]]
% TENSORBOX, 2018
%
d = numel(A);
R = size(A{1},2);
G = cell(d,1);
% G{1} = reshape(A{1},1,[],R);G{d} = A{1}';
% for k = 2: d-1
%     G{k} = full(ktensor({eye(R),A{k},eye(R)}));
% end

G{1} = A{1};G{d} = A{end};
for k = 2: d-1
    G{k} = full(ktensor({A{k},eye(R),eye(R)}));
end

Y = tt_tensor(G);
