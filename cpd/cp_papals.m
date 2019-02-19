function [P,output] = cp_papals(X,R,delta,opts)
% Parallel implementation to seek a good partition of [1:R] for the PALS
% algorithm
%
% INPUT:
%   X:  N-way data which can be a tensor or a ktensor.
%   R:  rank of the approximate tensor
%   OPTS: optional parameters
%     .tol:    tolerance on difference in fit {1.0e-4}
%     .maxiters: Maximum number of iterations {200}
%     .init: Initial guess [{'random'}|'nvecs'| 'orthogonal'|'fiber'| ktensor| cell array]
%          init can be a cell array whose each entry specifies an intial
%          value. The algorithm will chose the best one after small runs.
%          For example,
%          opts.init = {'random' 'random' 'nvec'};
%     .printitn: Print fit every n iterations {1}
%     .fitmax
%     .TraceFit: check fit values as stoping condition.
%     .TraceMSAE: check mean square angular error as stoping condition
%
% OUTPUT:
%  P:  ktensor of estimated factors
%  output:
%      .Fit
%      .NoIters
%
% EXAMPLE
%   X = tensor(randn([10 20 30]));
%   opts = cp_fastals;
%   opts.init = {'nvec' 'random' 'random'};
%   [P,output] = cp_fastals(X,5,opts);
%   figure(1);clf; plot(output.Fit(:,1),1-output.Fit(:,2))
%   xlabel('Iterations'); ylabel('Relative Error')
%
% REF:
%
% See also: cp_pals
%
% TENSOR BOX, v1. 2012
% Copyright 2011, Phan Anh Huy.
% 2012, Fast CP gradient
% 2013, Fix fast CP gradients for sparse tensor
% 2013, Extend GRAM intialization for higher order CPD



%% Fill in optional variable
if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    P = param; return
end

if param.linesearch
    param.TraceFit = true;
end
%% Parameters for linesearch
param_ls = struct('acc_pow',2, ... % Extrapolate to the iteration^(1/acc_pow) ahead
    'acc_fail',0, ... % Indicate how many times acceleration have failed
    'max_fail',4); % Increase acc_pow with one after max_fail failure

%%
N = ndims(X); I = size(X);
if isempty(param.normX)
    normX = norm(X);
else
    normX = param.normX;
end

if isa(X,'tensor')
    IsReal = isreal(X.data);
elseif isa(X,'ktensor') || isa(X,'ttensor')
    IsReal = all(cellfun(@isreal,X.u));
end

%% Initialize factors U
param.cp_func = str2func(mfilename);
Uinit = cp_init(X,R,param); U = Uinit;

%% Output
if param.printitn ~=0
    fprintf('\nCP-PALS:\n');
end

if nargout >=2
    output = struct('Uinit',{Uinit},'NoIters',[]);
    if param.TraceFit
        output.Fit = [];
    end
    if param.TraceMSAE
        output.MSAE = [];
    end
end

%% Reorder tensor dimensions and update order of factor matrices
% Permute the tensor to an apprppriate model with low computational cost
 
%%
if param.verify_convergence == 1
    %lambda = ones(R,1);
    %P = ktensor(U);
    %err=normX.^2 + norm(P).^2 - 2 * innerprod(X,P);
    %fit = 1-sqrt(err)/normX;
    fit = 0;
    %if param.TraceFit
    %    output.Fit = fit;
    %end
    if param.TraceMSAE
        msae = (pi/2)^2;
    end
end


P = arrange(ktensor(U));
lambda= P.lambda;
U = P.U;

%% Main Loop: Iterate until convergence
Pmat = [];Pls = false;flagtol = 0;
lambda_ = lambda.';
cost_ = norm(lambda)^2;

for iter = 1:param.maxiters
    pre_cost = cost_(end);
    if param.verify_convergence==1
        if param.TraceFit, fitold = fit;end
    end
    % |y - P * theta|^2 = |y|^2 + theta'*P'*P * theta - 2 * (y'*P) * theta
    %    = |y|^2 + theta'* H  * theta - 2 * g' * theta
    %    = |y|^2 + (F*theta)'* (F*theta) - 2 * (inv(F')*g)' * F*theta
    %    = |(inv(F')*g) - F*theta|^2 + |y|^2 - |inv(F')*g|^2

    
%     % Iterate over all N modes of the tensor
%     [U,lambda] = inner_update(X,U,lambda,delta,normX);
    [U,lambda] = par_inner_update(X,U,lambda,delta,normX);
    lambda_ = [lambda_ ;lambda(:).'];
    cost_ = [cost_; norm(lambda)^2];

    
%     %%
%      % Iterate over all N modes of the tensor
%     for r = 1:R
%         i1= r;
%         i2 = [1:r-1 r+1:R];
%         i3 = [];
%         ord = 1:N;
%         for n = 1:N
%             ord = circshift(ord,1);
%             % there is no need to multiply lambda back to the factor
%             % matrices u1r and U2_i2 because the parameters will be
%             % estimated
%             %U{ord(1)}(:,r) =  U{ord(1)}(:,r)*lambda(r);
%             %U{ord(2)}(:,i2) =  bsxfun(@times,U{ord(2)}(:,i2),lambda(i2)');
%             
%             %[H,g]=gH(permute(X,ord),U(ord),i1,i2,i3);
%             [H,g]=gH2(permute(X,ord),U(ord),i1,i2,i3);
%             
%             F = chol(H);
%             b = inv(F')*g;
%             deltan = delta^2 - normX^2 + norm(b)^2;
%             
%             %err1 = norm(X - full(ktensor(U)))^2; = normX^2 + theta'*H*theta - 2*g'*theta;
%             %err2 = norm(b - F*theta)^2 + normX^2 - norm(b)^2;
%             if deltan>0
%                 deltan = sqrt(deltan);
%                 % MEthOD 1: full form, but expensive
%                 [Un,cone_d] = linreg_bound(b(:),F,deltan);                
%                 
%                 ur = Un(1:I(ord(1)));
%                 Ur = reshape(Un(I(ord(1))+1:end),[],R-1);
%                 
%                 lambda(r) = norm(ur);
%                 lambda(i2) = sqrt(sum(Ur.^2));
%                 U{ord(1)}(:,r) =  ur/lambda(r);
%                 U{ord(2)}(:,i2) =  bsxfun(@rdivide,Ur,lambda(i2)');
%             end
%             
%             lambda_ = [lambda_ ;lambda(:).'];
%             cost_ = [cost_; norm(lambda)^2];
%         end
%     end
    %%
%  ord = 1:N;
%  for n = 1:N
%      ord = circshift(ord,1);
%      % there is no need to multiply lambda back to the factor
%      % matrices u1r and U2_i2 because the parameters will be
%      % estimated
%      %U{ord(1)}(:,r) =  U{ord(1)}(:,r)*lambda(r);
%      %U{ord(2)}(:,i2) =  bsxfun(@times,U{ord(2)}(:,i2),lambda(i2)');
%      
%      ii = randperm(R);
%      i1 = ii(1:ceil(R/3));
%      i2 = ii(ceil(R/3)+1:2*ceil(R/3));
%      i3 = ii(2*ceil(R/3)+1:end);
%      [H,g]=gH(permute(X,ord),U(ord),i1,i2,i3);
%      %[H,g]=gH2(permute(X,ord),U(ord),i1,i2,i3);
%      
%      F = chol(H);
%      b = inv(F')*g;
%      deltan = delta^2 - normX^2 + norm(b)^2;
%      
%      if deltan>0
%          deltan = sqrt(deltan);
%          % MEthOD 1: full form, but expensive
%          [Un,cone_d] = linreg_bound(b(:),F,deltan);
%          
%          un = vec_to_mat(Un,i1,i2,i3,I(ord));
%          
%          lambda(i1) = sqrt(sum(un{1}.^2));
%          U{ord(1)}(:,i1) = bsxfun(@rdivide,un{1},lambda(i1)');
%          lambda(i2) = sqrt(sum(un{2}.^2));
%          U{ord(2)}(:,i2) = bsxfun(@rdivide,un{2},lambda(i2)');
%          lambda(i3) = sqrt(sum(un{3}.^2));
%          U{ord(3)}(:,i3) = bsxfun(@rdivide,un{3},lambda(i3)');
%      end
%      
%      % cost = |lambda|^2
%      lambda_ = [lambda_ ;lambda(:).'];
%      cost_ = [cost_; norm(lambda)^2];
%      
%  end 
    
    if param.verify_convergence==1
        %if param.TraceFit
        P = ktensor(lambda,U);
        normresidual = sqrt(normX^2 + norm(P)^2 - 2 * real(innerprod((P),X)));
          
        fit = 1 - (normresidual/ normX); %fraction explained by model
        
        %fitchange = abs(fitold - fit);
        cost_diff = abs(cost_(end)-pre_cost)/pre_cost;
        %cost_diff = abs(cost_(end)-cost_(end-N))/cost_(end-N);
        if (iter > 1) && (cost_diff < param.tol) % Check for convergence
            flagtol = flagtol + 1;
        else
            flagtol = 0;
        end
        stop(1) = flagtol >= 10;
        
        
        %stop(3) = fit >= param.fitmax;
        
        
        if nargout >=2
            output.Fit = [output.Fit; iter fit];
        end
 
        
        if mod(iter,param.printitn)==0
            fprintf(' Iter %2d: ',iter);
            fprintf('|lambda|^2 = %e diff = %7.1e ', cost_(end), cost_diff);
            fprintf('\n');
        end
        
        % Check for convergence
        if (iter > 1) && any(stop)
            break;
        end
    end
end

%% Clean up final result
% Convert conj(U{n}) back to U{n}
if ~IsReal % complex conjugate of U{n} for fast CP-gradient
    U = cellfun(@conj,U,'uni',0);
end

% Arrange the final tensor so that the columns are normalized.
P = ktensor(lambda,U);

% Normalize factors and fix the signs
P = arrange(P);P = fixsigns(P);

% Display final fit
if param.printitn>0
    %innXXhat = sum(sum(U{updateorder(end)}.*conj(G)));
    %normresidual = sqrt(normX^2 + sum(sum(prod(UtU,3))) - 2*real(innXXhat));
    normresidual = sqrt(normX^2 + norm(P)^2 - 2 * real(innerprod(X,(P)) ));
    fit = 1 - (normresidual / normX); %fraction explained by model
    fprintf('Final fit = %e \n', fit);
end
 
if nargout >=2
    output.NoIters = iter;
    output.cost = cost_;
    output.lambda = lambda_;
end
end


function [U,lambda] = par_inner_update(X,U,lambda0,delta,normX)

R = size(U{1},2);
I = size(X);N = numel(I);
lambda = lambda0;
maxinnersearch = 20;
un_ = cell(maxinnersearch,1);
parfor kinner = 1:maxinnersearch
    % |y - P * theta|^2 = |y|^2 + theta'*P'*P * theta - 2 * (y'*P) * theta
    %    = |y|^2 + theta'* H  * theta - 2 * g' * theta
    %    = |y|^2 + (F*theta)'* (F*theta) - 2 * (inv(F')*g)' * F*theta
    %    = |(inv(F')*g) - F*theta|^2 + |y|^2 - |inv(F')*g|^2
    if kinner == 1
        ii = 1:R;
        iparts = {ii [] []};
    
    elseif kinner == 2
        ii = 1:R;
        iparts = {[] ii  []};
    
    elseif kinner == 3
        ii = 1:R;
        iparts = {[] [] ii};

    elseif kinner == 4        
        [ldas,ill] = sort(lambda0,'descend');
        % reshape(ldas(1:3*floor(R/3)),3,[]);
        ii = reshape(ill(1:3*floor(R/3)),3,[])';
        ii = [ii(:)' numel(ii)+1:R];
        iparts = mat2cell(ii,1,[ceil(R/3),ceil(R/3),R-2*ceil(R/3)]);
    else
        ii = randperm(R);
        iparts = mat2cell(ii,1,[ceil(R/3),ceil(R/3),R-2*ceil(R/3)]);
    end
    
    % construct the Hessian and gradient 
    [H,g]=gH(X,U,iparts{1},iparts{2},iparts{3});
    %[H,g]=gH2(X,U,i1,i2,i3);
    
    try
        F = chol(H);
    catch 
        continue
    end
    b = (F')\g;
    deltan = delta^2 - normX^2 + norm(b)^2;
    
    if deltan>0
        deltan = sqrt(deltan);
        % MEthOD 1: full form, but expensive
        [un,cone_d] = linreg_bound(b(:),F,deltan);
        
        un_{kinner} = vec_to_mat(un,iparts{1},iparts{2},iparts{3},I);
        iparts_{kinner} = iparts;
    end    
end

best_err = 0;un_best =[];
for kinner = 1:maxinnersearch
    if ~isempty(un_{kinner})
        un = un_{kinner};
        iparts = iparts_{kinner};
        for kn = 1:N
            lambda(iparts{kn}) = sqrt(sum(un{kn}.^2));
        end
        err = norm(lambda0)^2 - norm(lambda)^2;
        if err>best_err
            best_err = err;
            best_ii = kinner;
            best_iparts = iparts;
            lambda_best = lambda;
            for kn = 1:N
                if ~isempty(iparts{kn})
                    un{kn} = bsxfun(@rdivide,un{kn},lambda(iparts{kn})');
                end
            end
            un_best = un;
        end
    end
end

if ~isempty(un_best)
    un = un_best;
    lambda = lambda_best;
    fprintf('Best partition %d %s\n',best_ii, sprintf('%d',cell2mat(best_iparts)))
    for kn = 1:3
        U{kn}(:,best_iparts{kn}) = un{kn};
    end
end
end



function [U,lambda] = inner_update(X,U,lambda0,delta,normX)

R = size(U{1},2);
I = size(X);
best_err = 0;
lambda = lambda0;
maxinnersearch = 20;
un_best = [];
for kinner = 1:maxinnersearch
    % |y - P * theta|^2 = |y|^2 + theta'*P'*P * theta - 2 * (y'*P) * theta
    %    = |y|^2 + theta'* H  * theta - 2 * g' * theta
    %    = |y|^2 + (F*theta)'* (F*theta) - 2 * (inv(F')*g)' * F*theta
    %    = |(inv(F')*g) - F*theta|^2 + |y|^2 - |inv(F')*g|^2
    if kinner == 1
        ii = 1:R;
        iparts = {ii [] []};
    
    elseif kinner == 2
        ii = 1:R;
        iparts = {[] ii  []};
    
    elseif kinner == 3
        ii = 1:R;
        iparts = {[] [] ii};

    elseif kinner == 4        
        [ldas,ill] = sort(lambda0,'descend');
        % reshape(ldas(1:3*floor(R/3)),3,[]);
        ii = reshape(ill(1:3*floor(R/3)),3,[])';
        ii = [ii(:)' numel(ii)+1:R];
        iparts = mat2cell(ii,1,[ceil(R/3),ceil(R/3),R-2*ceil(R/3)]);
    else
        ii = randperm(R);
        iparts = mat2cell(ii,1,[ceil(R/3),ceil(R/3),R-2*ceil(R/3)]);

    end
    
    % construct the Hessian and gradient 
    [H,g]=gH(X,U,iparts{1},iparts{2},iparts{3});
    %[H,g]=gH2(X,U,i1,i2,i3);
    
    try
        F = chol(H);
    catch 
        continue
    end
    b = (F')\g;
    deltan = delta^2 - normX^2 + norm(b)^2;
    
    if deltan>0
        deltan = sqrt(deltan);
        % MEthOD 1: full form, but expensive
        [un,cone_d] = linreg_bound(b(:),F,deltan);
        
        un = vec_to_mat(un,iparts{1},iparts{2},iparts{3},I);
        
        for kn = 1:3
            lambda(iparts{kn}) = sqrt(sum(un{kn}.^2));
        end
        err = norm(lambda0)^2 - norm(lambda)^2;
        if err>best_err
            best_err = err;
            best_ii = kinner;
            best_iparts = iparts;
            lambda_best = lambda;
            for kn = 1:3
                if ~isempty(iparts{kn})
                    un{kn} = bsxfun(@rdivide,un{kn},lambda(iparts{kn})');
                end
            end
            un_best = un;
        end
    end    
end
if ~isempty(un_best)
    un = un_best;
    lambda = lambda_best;
    fprintf('Best partition %d %s\n',best_ii, sprintf('%d',cell2mat(best_iparts)))
    for kn = 1:3
        U{kn}(:,best_iparts{kn}) = un{kn};
    end
end
end

function un = vec_to_mat(v,i1,i2,i3,sz)
% 
% v is a vector v = [vec(U1); vec(U2) ; vec(U{3})]
un = mat2cell(v,sz(:).*[numel(i1); numel(i2); numel(i3)]);
for kn = 1:numel(un)
    un{kn} = reshape(un{kn},sz(kn),[]);
end
end

function  [H,g,theta]=gH2(T,U,i1,i2,i3)
%% gradient and Hessian for A partition [i1,I2 ,I3]
[A,B,C] = deal(U{1},U{2},U{3});
[n,r]=size(A);
sz(1) = n;
sz(2) = size(B,1);
ii = [i1 i2 i3];
A = A(:,ii);
B = B(:,ii);
C = C(:,ii);

i1 = 1:numel(i1);
i2 = numel(i1)+1:numel(i1)+numel(i2);
i3 = numel(i1)+numel(i2)+1:r;

if nargout>2
    theta = [reshape(A(:,i1),[],1); reshape(B(:,i2),[],1); reshape(C(:,i3),[],1)];
end

gamma_1 = (B(:,i1)'*B(:,i1)).*(C(:,i1)'*C(:,i1)); % is always 1 because of normalizatiob
% gamma_1 is always one because of normalization 
Gamma_2 = (A(:,i2)'*A(:,i2)).*(C(:,i2)'*C(:,i2));
gamma_3 = C(:,i2)'*C(:,i1);

H21 = kron(diag(gamma_3)*A(:,i2)',B(:,i1));
H = [gamma_1*eye(sz(1)) H21'
    H21 kron(Gamma_2,eye(sz(2)))];

g = [double(ttv(T,{B(:,i1),C(:,i1)},[2 3]))
    reshape(mttkrp(T,{A(:,i2) B(:,i2),C(:,i2)},2),[],1)];
end
% 
function [H,g,theta]=gH(T,U,i1,i2,i3)
% gradient and Hessian for arbitrary partition [i1,I2 ,I3]
%
%% Reorder columns of A, B and C such that i1 < i2 < i3
[A,B,C] = deal(U{1},U{2},U{3});
[na,r]=size(A);
[nb,r]=size(B);
[nc,r]=size(C);
n = [na nb nc];

ii = [i1 i2 i3];
inum = [numel(i1) numel(i2) numel(i3)];

A = A(:,ii);
B = B(:,ii);
C = C(:,ii);

i1 = 1:numel(i1);
i2 = numel(i1)+1:numel(i1)+numel(i2);
i3 = numel(i1)+numel(i2)+1:r;

if nargout>2
    theta = [reshape(A(:,i1),[],1); reshape(B(:,i2),[],1); reshape(C(:,i3),[],1)];
end

%%
J=zeros(na*nb*nc,n*inum(:));
ic=length(i1);
if ic>0
    J(:,1:ic*na)=kron(khatrirao(C(:,i1),B(:,i1)),eye(na));
end
ic2=length(i2);
if ic2>0
    ipoc = ic*na;
    for ir=1:ic2
        J(:,ipoc+1:ipoc+nb)=kron(C(:,i2(ir)),kron(eye(nb),A(:,i2(ir))));
        ipoc=ipoc+ nb;
    end
end

ic3=length(i3);
if ic3>0
    ipoc = ic*na+ic2*nb;
    for ir=1:ic3
        J(:,ipoc+1:ipoc+nc)=kron(eye(nc),kron(B(:,i3(ir)),A(:,i3(ir))));
        ipoc=ipoc+nc;
    end
end
H=J'*J;
g = J'*T(:);
end

%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function [K,K2] = mkhatrirao(A)
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
function [K,K2] = khatrirao_r2l(A)
% Fast Khatri-Rao product
% P = fastkhatrirao({A,B,C})
%
%Copyright 2012, Phan Anh Huy.

R = size(A{1},2);
K = A{end};
if nargout == 1
    for i = numel(A)-1:-1:1
        K = bsxfun(@times,reshape(K,[],1,R),reshape(A{i},1,[],R));
    end
elseif numel(A) > 2
    for i = numel(A)-1:-1:2
        K = bsxfun(@times,reshape(K,[],1,R),reshape(A{i},1,[],R));
    end
    K2 = reshape(K,[],R);
    K = bsxfun(@times,reshape(K,[],1,R),reshape(A{1},1,[],R));
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
K = A{1}.';

for i = 2:numel(A)
    K = bsxfun(@times,reshape(A{i}.',R,[]),reshape(K,R,1,[]));
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
param.addOptional('linesearch',true);
%param.addParamValue('verify_convergence',true,@islogical);
param.addParamValue('TraceFit',true,@islogical);
param.addParamValue('TraceMSAE',false,@islogical);

param.addOptional('normX',[]);

param.parse(opts);
param = param.Results;
param.verify_convergence = param.TraceFit || param.TraceMSAE;
end


%%

function [Pnew,param_ls] = cp_linesearch(X,P,U0,param_ls)
% Simple line search adapted from Rasmus Bro's approach.

alpha = param_ls.alpha^(1/param_ls.acc_pow);
U = P.U;
U{1} = U{1} * diag(P.lambda);

Unew = cellfun(@(u,uold) uold + (u-uold) * alpha,U,U0,'uni',0);
Pnew = ktensor(Unew);
mod_newerr = norm(Pnew)^2 - 2 * innerprod(X,Pnew);

if mod_newerr>param_ls.mod_err
    param_ls.acc_fail=param_ls.acc_fail+1;
    Pnew = false;
    if param_ls.acc_fail==param_ls.max_fail,
        param_ls.acc_pow=param_ls.acc_pow+1+1;
        param_ls.acc_fail=0;
    end
else
    param_ls.mod_err = mod_newerr;
end

end

%%

function [msae,msae2,sae,sae2] = SAE(U,Uh)
% Square Angular Error
% sae: square angular error between U and Uh
% msae: mean over all components
%
% [1] P. Tichavsky and Z. Koldovsky, Stability of CANDECOMP-PARAFAC
% tensor decomposition, in ICASSP, 2011, pp. 4164?4167.
%
% [2] P. Tichavsky and Z. Koldovsky, Weight adjusted tensor method for
% blind separation of underdetermined mixtures of nonstationary sources,
% IEEE Transactions on Signal Processing, 59 (2011), pp. 1037?1047.
%
% [3] Z. Koldovsky, P. Tichavsky, and A.-H. Phan, Stability analysis and fast
% damped Gauss-Newton algorithm for INDSCAL tensor decomposition, in
% Statistical Signal Processing Workshop (SSP), IEEE, 2011, pp. 581?584.
%
% Phan Anh Huy, 2011

N = numel(U);
R = size(U{1},2);
sae = zeros(N,size(Uh{1},2));
sae2 = zeros(N,R);
for n = 1: N
    C = U{n}'*Uh{n};
    C = C./(sqrt(sum(abs(U{n}).^2))'*sqrt(sum(abs(Uh{n}).^2)));
    C = acos(min(1,abs(C)));
    sae(n,:) = min(C,[],1).^2;
    sae2(n,:) = min(C,[],2).^2;
end
msae = mean(sae(:));
msae2 = mean(sae2(:));
end

