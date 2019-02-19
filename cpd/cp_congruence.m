function C = cp_congruence(P)
% Compute congruence coefficients for the Kruskal tensor P
%
% TENSORBOX, 2018
P = arrange(normalize(P));
R = size(P.u{1},2);
C = prod(reshape(cell2mat(cellfun(@(x) x'*x,P.u(:),'uni',0))',R,R,[]),3);
end