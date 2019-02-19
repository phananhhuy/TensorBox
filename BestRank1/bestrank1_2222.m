function [P,cost] = bestrank1_2222(Y,opts)
% Find best rank-1 tensor approximation to the tensor Y of size 2x2x2x2
% The algorithm exploits the closed-form best rank-1 tensor approximation
% to an order-3 tensor
%
% cost = Y x {u}
%
% Phan Anh-Huy, 2017
% August 16, 2017
%

if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    P = param; return
end
N = ndims(Y);
U = cp_init(Y,1,param);
P = normalize(ktensor(U));
U = P.u;

cost = zeros(param.maxiters,N); % Y x {u_n}

for kiter = 1:param.maxiters
    
    for n = 1:N
        Y4 = ttv(Y,U,n);
        m = [1:n-1 n+1:N];
        [U{m(1)},U{m(2)},U{m(3)},smax] = bestrank1_222(double(Y4));
        
        t4 = double(ttv(Y,U,-n));
        cost(kiter,n) = norm(t4);
        U{n} = t4/cost(kiter,n);
    end
    
    if mod(kiter,param.printitn)==0
        fprintf(' Iter %2d: ',kiter);
        %         if param.TraceFit
        if kiter>1
            dchange = cost(kiter-1)-cost(kiter);
        else
            dchange = 0;
        end
        fprintf('Cost = %e delta = %7.1e ', cost(kiter), dchange);
        %         end
        fprintf('\n');
    end
    
    if (kiter>2) && abs(cost(kiter)-cost(kiter-1))< param.tol
        break
    end
end
cost = cost(1:kiter,:);
P = ktensor(cost(end),U);


%%
% [v3d,v4d] = meshgrid(6:-1:0,6:-1:0);
% [A,B,C1,C2] = gen_poly(Y);
% % sol=polynsolve({[C1(:) v4d(:) v3d(:)]  [C2(:) v4d(:) v3d(:)]});
% 
% Eq2 = '';
% Eq1 = '';
% 
% for k = 1:49
%     Eq1 = [Eq1 ' + ' sprintf('(%d) * x1^%d * x2^%d ',C1(k),v3d(k),v4d(k))];
%     Eq2 = [Eq2 ' + ' sprintf('(%d) * x1^%d * x2^%d ',C2(k),v3d(k),v4d(k))];
% end
%   
% %%
% [PowersMat, CoeffsArray, MaxPowersArray] = IritEquationParser(2, {Eq1 Eq2});
% ii2 = sub2ind([7 7],OutPowersMat{1}(1,:)'+1,OutPowersMat{1}(2,:)'+1);
% [~,iij] = sort(ii2);
% 
% CoeffsArray{1}(iij) = C1(:)';
% CoeffsArray{2}(iij) = C2(:)';
% 
% % Call mex function to receive solution. %
% Properties.LowerBounds = [v3v-5 v4v-5];
% Properties.UpperBounds = [v3v+5 v4v+5];
% Properties.Constraints = zeros(1,2);
% Properties.NumericTol =  1e-15;
% Properties.SubdivTol = 1e-3;
% Properties.Step = 1e-5;
% 
% [Sol, T] = IritPolynomialSolverMex(2, PowersMat, CoeffsArray, MaxPowersArray, Properties.LowerBounds, Properties.UpperBounds, Properties.Constraints, Properties.NumericTol, Properties.SubdivTol, Properties.Step);
% % 
% % fsigma_sv = zeros(size(Sol,1),1);
% % for ks = 1:size(Sol,1)
% %     v3s_num = Sol(ks,1);
% %     v4s_num = Sol(ks,2);
% %     fsigma_sv(ks) = double(subs(fsigma,[st; sw; x3;x4],[tyv; wyv; atan(v3s_num) ; atan(v4s_num)]));
% % end
%%
% error
% error = norm(Y)^2 - cost.^2;
end



%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init','random',@(x) (iscell(x) || isa(x,'ktensor')||...
    ismember(x(1:4),{'rand' 'nvec' 'fibe' 'orth' 'dtld'})));
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('fitmax',1-1e-10);

param.parse(opts);
param = param.Results;
end


function [A,B,C1,C2] = gen_poly(y)

% generate two matrices of size 3x3 ,  A and B

tc3c4 = 1/4* (y(1)^2 + y(2)^2 + y(3)^2 + y(4)^2 - y(5)^2 - y(6)^2 - y(7)^2 - y(8)^2 - y(9)^2 - y(10)^2 - y(11)^2 - y(12)^2 + y(13)^2 + y(14)^2 + y(15)^2 + y(16)^2);
tc3s4 = 1/2*(y(1)*y(9) + y(2)*y(10) + y(3)*y(11) + y(4)*y(12) - y(5)*y(13) - y(6)*y(14) - y(7)*y(15) - y(8)*y(16));
ts3c4 = 1/2*(y(1)*y(5) + y(2)*y(6) + y(3)*y(7) + y(4)*y(8) - y(9)*y(13) - y(10)*y(14) - y(11)*y(15) - y(12)*y(16));
ts3s4 = 1/2*(y(1)*y(13) + y(5)*y(9) + y(2)*y(14) + y(6)*y(10) + y(3)*y(15) + y(7)*y(11) + y(4)*y(16) + y(8)*y(12));
ts3 = 1/2*(y(1)*y(5) + y(2)*y(6) + y(3)*y(7) + y(4)*y(8) + y(9)*y(13) + y(10)*y(14) + y(11)*y(15) + y(12)*y(16));
ts4 = 1/2*(y(1)*y(9) + y(2)*y(10) + y(3)*y(11) + y(4)*y(12) + y(5)*y(13) + y(6)*y(14) + y(7)*y(15) + y(8)*y(16));
tc3 = 1/4*(y(1)^2 + y(2)^2 + y(3)^2 + y(4)^2 - y(5)^2 - y(6)^2 - y(7)^2 - y(8)^2 + y(9)^2 + y(10)^2 + y(11)^2 + y(12)^2 - y(13)^2 - y(14)^2 - y(15)^2 - y(16)^2);
tc4 = 1/4*(y(1)^2 + y(2)^2 + y(3)^2 + y(4)^2 + y(5)^2 + y(6)^2 + y(7)^2 + y(8)^2 - y(9)^2 - y(10)^2 - y(11)^2 - y(12)^2 - y(13)^2 - y(14)^2 - y(15)^2 - y(16)^2);
t00 = 1/4*(y(1)^2 + y(2)^2 + y(3)^2 + y(4)^2 + y(5)^2 + y(6)^2 + y(7)^2 + y(8)^2 + y(9)^2 + y(10)^2 + y(11)^2 + y(12)^2 + y(13)^2 + y(14)^2 + y(15)^2 + y(16)^2);


A = reshape([tc3c4
    ts3c4
    tc4
    tc3s4
    ts3s4
    ts4
    tc3
    ts3
    t00],3,3);


wc3c4 = (y(1)*y(4) - y(2)*y(3) - y(5)*y(8) + y(6)*y(7) - y(9)*y(12) + y(10)*y(11) + y(13)*y(16) - y(14)*y(15))/2;
wc3s4 = (y(1)*y(12) - y(2)*y(11) - y(3)*y(10) + y(4)*y(9) - y(5)*y(16) + y(6)*y(15) + y(7)*y(14) - y(8)*y(13))/2;
ws3c4 = (y(1)*y(8) - y(2)*y(7) - y(3)*y(6) + y(4)*y(5) - y(9)*y(16) + y(10)*y(15) + y(11)*y(14) - y(12)*y(13))/2;
ws3s4 = (y(1)*y(16) - y(2)*y(15) - y(3)*y(14) + y(4)*y(13) + y(5)*y(12) - y(6)*y(11) - y(7)*y(10) + y(8)*y(9))/2;
wc3 = (y(1)*y(4) - y(2)*y(3) - y(5)*y(8) + y(6)*y(7) + y(9)*y(12) - y(10)*y(11) - y(13)*y(16) + y(14)*y(15))/2;
wc4 = (y(1)*y(4) - y(2)*y(3) + y(5)*y(8) - y(6)*y(7) - y(9)*y(12) + y(10)*y(11) - y(13)*y(16) + y(14)*y(15))/2;
ws3 = (y(1)*y(8) - y(2)*y(7) - y(3)*y(6) + y(4)*y(5) + y(9)*y(16) - y(10)*y(15) - y(11)*y(14) + y(12)*y(13))/2;
ws4 = (y(1)*y(12) - y(2)*y(11) - y(3)*y(10) + y(4)*y(9) + y(5)*y(16) - y(6)*y(15) - y(7)*y(14) + y(8)*y(13))/2;
w00 = (y(1)*y(4) - y(2)*y(3) + y(5)*y(8) - y(6)*y(7) + y(9)*y(12) - y(10)*y(11) + y(13)*y(16) - y(14)*y(15))/2;


B = reshape([wc3c4
    ws3c4
    wc4
    wc3s4
    ws3s4
    ws4
    wc3
    ws3
    w00],3,3);

% generate two matrices of size 7x7 ,  C1 and C2 as coefficients of bi-variate polynomials
% derived from the gradients
%
%   gn = sum_{k,l}  Cn(k,l) * v3^(7-k) v4^(7-l)
% Ka = Kp*Kp2*Kp3;
K = [
    -1     0     3     0    -3     0     1
    0     2     0    -4     0     2     0
    1     0    -1     0    -1     0     1
    0     2     0    -4     0     2     0
    0     0    -4     0     4     0     0
    0    -2     0     0     0     2     0
    1     0    -1     0    -1     0     1
    0    -2     0     0     0     2     0
    -1     0    -1     0     1     0     1
    0     2     0    -4     0     2     0
    0     0    -4     0     4     0     0
    0    -2     0     0     0     2     0
    0     0    -4     0     4     0     0
    0     0     0     8     0     0     0
    0     0     4     0     4     0     0
    0    -2     0     0     0     2     0
    0     0     4     0     4     0     0
    0     2     0     4     0     2     0
    1     0    -1     0    -1     0     1
    0    -2     0     0     0     2     0
    -1     0    -1     0     1     0     1
    0    -2     0     0     0     2     0
    0     0     4     0     4     0     0
    0     2     0     4     0     2     0
    -1     0    -1     0     1     0     1
    0     2     0     4     0     2     0
    1     0     3     0     3     0     1];

F = [ 0 -1 0
    1 0 0
    0 0 0];

% tic
C1 = K'*(kron(B,kron(F'*A,F'*A)+kron(F'*B,F'*B)) - 2 * kron(A,kron(F'*A,F'*B)))*K;
C2 = K'*(kron(B,kron(A*F,A*F)+kron(B*F,B*F)) - 2 * kron(A,kron(A*F,B*F)))*K;
% toc
end