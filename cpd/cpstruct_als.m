function [P,output] = cpstruct_fastals(Fact,param_struct,R,opts)
% Fast ALS for CP factorizes the N-way tensor X into factors of R components.
% The fast CP ALS was adopted from the CP_ALS algorithm [2]
% to employ the fast computation of CP gradients.
%
% INPUT:
%   X:  N-D data which can be a tensor or a ktensor.
%   R:  rank of the approximate tensor
%   OPTS: optional parameters
%     .tol:    tolerance on difference in fit {1.0e-4}
%     .maxiters: Maximum number of iterations {200}
%     .init: Initial guess [{'random'}|'nvecs'|cell array]
%     .printitn: Print fit every n iterations {1}
%     .fitmax
%     .TraceFit: check fit values as stoping condition.
%     .TraceMSAE: check mean square angular error as stoping condition
% Output:
%  P:  ktensor of estimated factors
%
% REF:
% [1] A.-H. Phan, P. Tichavsky, A. Cichocki, "On Fast Computation of Gradients
% for CP Algorithms", 2011
% [2] Matlab Tensor toolbox by Brett Bader and Tamara Kolda
% http://csmr.ca.sandia.gov/~tgkolda/TensorToolbox.
%
% See also: cp_als
%
% TENSOR BOX, v1. 2012
% Copyright 2011, Phan Anh Huy.
% 2012, Fast CP gradient

% param_struct.M = M indicator matrix
% param_struct.J = [J1 J2 ... Jm]
% param_struct.mode = m modes to have collinear components by M

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
N = numel(Fact);
dimorder = 1:N;
inmodes = param_struct.inmodes;
Jr = param_struct.Jr;       % Rin x 2 indicates number of replication of a component
cJr = [0;cumsum(Jr(:))];


outmodes = setdiff(dimorder,inmodes);

Rin = size(Fact{inmodes(1)},2);
Rout = size(Fact{outmodes(1)},2);   %% = SUM(JR)
Cin = zeros(Rin,Rin,numel(inmodes));
for k = 1:numel(inmodes)
    n = inmodes(k);
    Cin(:,:,k) = Fact{n}'*Fact{n};
end
Cout = zeros(Rout,Rout,numel(outmodes));
for k = 1:numel(outmodes)
    n = outmodes(k);
    Cout(:,:,k) = Fact{n}'*Fact{n};
end
% M = [];
repidx = zeros(Rout,1);
for r = 1:Rin
%     M = blkdiag(M, ones(1,Jr(r)));
    repidx(cJr(r)+1:cJr(r+1)) = r;
end

% % Ktensor Fact
% V = Fact;
% for k = 1:numel(inmodes)
%     n = inmodes(k);
%     V{n} = V{n} * M;
% end
% X = ktensor(V);
%normX = norm(X);
%normX = sqrt(sum(sum(M'*prod(Cin,3)*M .* prod(Cout,3))));
ss = zeros(Rin,Rout);
Gammain = prod(Cin,3);
Gammaout = prod(Cout,3);
for r = 1:Rin
    ss(r,:) = sum(Gammaout(cJr(r)+1:cJr(r+1),:),1);
end
ss2 = zeros(Rin,Rin);
for r = 1:Rin
    ss2(:,r) = sum(ss(:,cJr(r)+1:cJr(r+1)),2);
end
normX = sqrt(Gammain(:).' * ss2(:));
% I = size(X);
%% Initialize factors U
fprintf('\nFast structured CP_ALS:\n');
%Uinit = cp_init(X,R,param); U = Uinit;
Uinit = cpstruct_init(Fact,param_struct,R,param); U = Uinit;

%% Output
if nargout >=2
    output = struct('Uinit',{Uinit},'NoIters',[]);
    if param.TraceFit
        output.Fit = [];
    end
    if param.TraceMSAE
        output.MSAE = [];
    end
end

%%
UtU = zeros(R,R,N);
for n = 1:N
    UtU(:,:,n) = U{n}'*U{n};
end

FtUin = zeros(Rin,R,numel(inmodes));
for k = 1:numel(inmodes)
    n = inmodes(k);
    FtUin(:,:,k) = Fact{n}.'*conj(U{n});
end
FtUout = zeros(Rout,R,numel(outmodes));
for k = 1:numel(outmodes)
    n = outmodes(k);
    FtUout(:,:,k) = Fact{n}.'*conj(U{n});
end

if param.verify_convergence == 1
    lambda = ones(R,1);
    %P = ktensor(U);
    Gammaout = prod(FtUout,3); Gammain = prod(FtUin,3);
    ss = zeros(Rin,R);
    for r = 1:Rin
        ss(r,:) = sum(Gammaout(cJr(r)+1:cJr(r+1),:),1);
    end
    innerprd = Gammain(:).' * ss(:);
    normP2 = sum(sum(prod(UtU,3)));
    err =  (normX^2 + normP2 - 2 * real(innerprd));
    fit = 1-sqrt(err)/normX;
    if param.TraceFit
        output.Fit = fit;
    end
    if param.TraceMSAE
        msae = (pi/2)^2;
    end
end

%% Main Loop: Iterate until convergence
Pls = false;
for iter = 1:param.maxiters
    
    if param.verify_convergence==1
        if param.TraceFit, fitold = fit;end
        if param.TraceMSAE, msaeold = msae;end
    end
    
    if (param.verify_convergence==1) || (param.linesearch == true)
        Uold = U;
    end
    
    % Iterate over all N modes of the tensor
    Gammaout= prod(FtUout,3);
    Temp = zeros(Rin,R);
    for r = 1:Rin
        idr = cJr(r)+1:cJr(r+1);
        Temp(r,:) = sum(Gammaout(idr,:),1);
    end
    
    for kn = 1:numel(inmodes)
        n = inmodes(kn);
        
        % Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
        Gammain_n = prod(FtUin(:,:,[1:kn-1 kn+1:end]),3);
        
        %G2 = mttkrp(X,U,n);norm(G-G2,'fro')
        % Compute the matrix of coefficients for linear system
        %U{n} = Fact{n} * ((Gammain_n .* Temp)/prod(UtU(:,:,[1:n-1 n+1:N]),3));
        U{n} = Fact{n} * ((Gammain_n .* Temp)*pinv(prod(UtU(:,:,[1:n-1 n+1:N]),3).'));
        
        % Normalize each vector to prevent singularities in coefmatrix
        if iter == 1
            lambda = sqrt(sum(abs(U{n}).^2,1)).'; %2-norm % fixed for complex tensors
        else
            lambda = max( max(abs(U{n}),[],1), 1 ).'; %max-norm % fixed for complex tensors
        end
        U{n} = bsxfun(@rdivide,U{n},lambda.');
        UtU(:,:,n) = U{n}'*U{n};
        FtUin(:,:,kn) = Fact{n}.'*conj(U{n});
    end
    
    Gammain = prod(FtUin,3);
    Gammain = Gammain(repidx,:);
    for kn = 1:numel(outmodes)
        n = outmodes(kn);
        Gammaout_n = prod(FtUout(:,:,[1:kn-1 kn+1:end]),3);
        
%         for r = 1:Rin
%             idr = cJr(r)+1:cJr(r+1);
%             Gammaout_n(idr,:) = bsxfun(@times,Gammaout_n(idr,:),Gammain(r,:));
%         end
        
        %G2 = mttkrp(X,U,n);norm(G-G2,'fro')
        
        % Compute the matrix of coefficients for linear system
        U{n} = (Fact{n} * (Gammaout_n .* Gammain))/prod(UtU(:,:,[1:n-1 n+1:N]),3).';
        %U{n} = (pinv(prod(UtU(:,:,[1:n-1 n+1:N]),3)) * G')';
        
        % Normalize each vector to prevent singularities in coefmatrix
        if iter == 1
            lambda = sqrt(sum(abs(U{n}).^2,1)).'; %2-norm % fixed for complex tensors
        else
            lambda = max(max(abs(U{n}),[],1), 1 ).'; %max-norm % fixed for complex tensors
        end
        U{n} = bsxfun(@rdivide,U{n},lambda.');
        UtU(:,:,n) = U{n}'*U{n};
        FtUout(:,:,kn) = Fact{n}.'*conj(U{n});
    end
    U{n} = bsxfun(@times,U{n},lambda.');lambda = ones(R,1);
    UtU(:,:,n) = U{n}'*U{n};
    FtUout(:,:,kn) = Fact{n}.'*conj(U{n});
    
    if param.verify_convergence==1
        if param.TraceFit
            %P = ktensor(U);
            Gammaout = prod(FtUout,3); Gammain = prod(FtUin,3);
            ss = zeros(Rin,R);
            for r = 1:Rin
                ss(r,:) = sum(Gammaout(cJr(r)+1:cJr(r+1),:),1);
            end
            innerprd = Gammain(:).' * ss(:);
            normP2 = sum(sum(prod(UtU,3)));
            normresidual =  sqrt(normX^2 + normP2 - 2 * real(innerprd));
            fit = 1 - (normresidual/ normX); %fraction explained by model
            fitchange = abs(fitold - fit);
            stop(1) = fitchange < param.tol;
            stop(3) = fit >= param.fitmax;
            if nargout >=2
                output.Fit = [output.Fit; fit];
            end
        end
        
        if param.TraceMSAE
            msae = SAE(U,Uold);
            msaechange = abs(msaeold - msae); % SAE changes
            stop(2) = msaechange < param.tol*abs(msaeold);
            if nargout >=2
                output.MSAE = [output.MSAE; msae];
            end
        end
        
        if mod(iter,param.printitn)==0
            fprintf(' Iter %2d: ',iter);
            if param.TraceFit
                fprintf('fit = %e fitdelta = %7.1e ', fit, fitchange);
            end
            if param.TraceMSAE
                fprintf('msae = %e delta = %7.1e', msae, msaechange);
            end
            fprintf('\n');
        end
        
        % Check for convergence
        if (iter > 1) && any(stop)
            break;
        end
    end
end

%% Clean up final result
% Arrange the final tensor so that the columns are normalized.
P = ktensor(lambda,U);

% Normalize factors and fix the signs
P = arrange(P);P = fixsigns(P);

if param.printitn>0
    Gammaout = prod(FtUout,3); Gammain = prod(FtUin,3);
    ss = zeros(Rin,R);
    for r = 1:Rin
        ss(r,:) = sum(Gammaout(cJr(r)+1:cJr(r+1),:),1);
    end
    innerprd = Gammain(:).' * ss(:);
    normP2 = sum(sum(prod(UtU,3)));
    normresidual =  sqrt(normX^2 + normP2 - 2 * real(innerprd));
    fit = 1 - (normresidual / normX); %fraction explained by model
    fprintf(' Final fit = %e \n', fit);
end

if nargout >=2
    output.NoIters = iter;
end

end


% %% Khatri-Rao xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
% function krp = khatrirao(A,B)
% if nargin==2
%     R = size(A,2);
%     krp = zeros(size(A,1)*size(B,1),R);
%     for r = 1:R
%         d = B(:,r) * A(:,r)';
%         krp(:,r) = d(:);
%     end
% else
%
%     krp = A{1};
%     I = cellfun(@(x) size(x,1),A);
%     R = size(A{1},2);
%     for k = 2:numel(A)
%         temp = zeros(size(krp,1)*I(k),R);
%         for r = 1:R
%             d = A{k}(:,r) * krp(:,r)';
%             temp(:,r) = d(:);
%         end
%         krp = temp;
%     end
% end
% end
%% xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function K = khatrirao(A)
% Fast Khatri-Rao product
% P = fastkhatrirao({A,B,C})
%
%Copyright 2012, Phan Anh Huy.

R = size(A{1},2);
K = A{1};
for i = 2:numel(A)
    K = bsxfun(@times,reshape(A{i},[],1,R),reshape(K,1,[],R));
end
K = reshape(K,[],R);
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
param.addOptional('init','random',@(x) (iscell(x) || isa(x,'ktensor')||ismember(x(1:4),{'rand' 'nvec'})));
param.addOptional('maxiters',200);
param.addOptional('tol',1e-6);
param.addOptional('printitn',0);
param.addOptional('fitmax',1-1e-10);
param.addOptional('linesearch',true);
%param.addParamValue('verify_convergence',true,@islogical);
param.addParamValue('TraceFit',false,@islogical);
param.addParamValue('TraceMSAE',true,@islogical);

param.parse(opts);
param = param.Results;
param.verify_convergence = param.TraceFit || param.TraceMSAE;
end

%% Initialization xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function Uinit = cp_init(X,R,param)
% Set up and error checking on initial guess for U.
N = ndims(X);
if iscell(param.init)
    if (numel(param.init) == N) && all(cellfun(@isnumeric,param.init))
        Uinit = param.init;
        Sz = cell2mat(cellfun(@size,Uinit(:),'uni',0));
        if (numel(Uinit) ~= N) || (~isequal(Sz(:,1),size(X)')) || (~all(Sz(:,2)==R))
            error('Wrong Initialization');
        end
    else % small iteratons to find best initialization
        normX = norm(X);
        bestfit = 0;Pbest = [];
        for ki = 1:numel(param.init)
            initk = param.init{ki};
            if iscell(initk) || ...
                    (ischar(initk)  && ismember(initk(1:4),{'rand' 'nvec'}))  % multi-initialization
                cp_fun = str2func(mfilename);
                initparam = param;initparam.maxiters = 10;
                initparam.init = initk;
                P = cp_fun(X,R,initparam);
                fitinit = 1- sqrt(normX^2 + norm(P)^2 - 2 * real(innerprod(X,P)))/normX;
                if fitinit > bestfit
                    Pbest = P;
                    bestfit = fitinit;
                end
            end
        end
        Uinit = Pbest.U;
        Uinit{end} = bsxfun(@times,Uinit{end},Pbest.lambda(:)');
    end
elseif isa(param.init,'ktensor')
    Uinit = param.init.U;
    Uinit{end} = Uinit{end} * diag(param.init.lambda);
    Sz = cell2mat(cellfun(@size,Uinit,'uni',0));
    if (numel(Uinit) ~= N) || (~isequal(Sz(:,1),size(X)')) || (~all(Sz(:,2)==R))
        error('Wrong Initialization');
    end
elseif strcmp(param.init(1:4),'rand')
    Uinit = cell(N,1);
    for n = 1:N
        Uinit{n} = randn(size(X,n),R);
    end
elseif strcmp(param.init(1:4),'nvec')
    Uinit = cell(N,1);
    for n = 1:N
        if R < size(X,n)
            Uinit{n} = (nvecs(X,n,R));
        else
            Uinit{n} = randn(size(X,n),R);
        end
    end
else
    error('Invalid initialization');
end
end


%% Initialization xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function Uinit = cpstruct_init(Fact,param_struct,R,param)
% Set up and error checking on initial guess for U.
N = numel(Fact);
SzF = cell2mat(cellfun(@size,Fact(:),'uni',0));

if iscell(param.init)
    if (numel(param.init) == N) && all(cellfun(@isnumeric,param.init))
        Uinit = param.init;
        Sz = cell2mat(cellfun(@size,Uinit(:),'uni',0));
        if (numel(Uinit) ~= N) || (~isequal(Sz(:,1),SzF(:,1))) || (~all(Sz(:,2)==R))
            error('Wrong Initialization');
        end

    else % small iteratons to find the best initialization
        bestfit = -inf;Pbest = [];
        for ki = 1:numel(param.init)
            initk = param.init{ki};
            if iscell(initk) || isa(initk,'ktensor') || ...
                    (ischar(initk)  && ismember(initk(1:4), ...
                    {'rand' 'nvec' 'fibe' 'orth'}))  % multi-initialization
                if ischar(initk)
                    cprintf('blue','Init. %d - %s',ki,initk)
                else
                    cprintf('blue','Init. %d - %s',ki,class(initk))
                end
                
                cp_fun = str2func(mfilename);
                initparam = param;initparam.maxiters = 10;
                initparam.init = initk;
                [P,outinit] = cp_fun(Fact,param_struct,R,initparam);
                fitinit = outinit.Fit(end);
                if real(fitinit) > bestfit
                    Pbest = P;
                    bestfit = fitinit;
                end
            end
        end
        cprintf('blue','Choose the best initial value.\n')
        Uinit = Pbest.U;
        Uinit{end} = bsxfun(@times,Uinit{end},Pbest.lambda(:)');
     
    end
elseif isa(param.init,'ktensor')
    Uinit = param.init.U;
    Uinit{end} = Uinit{end} * diag(param.init.lambda);
    Sz = cell2mat(cellfun(@size,Uinit,'uni',0));
    if (numel(Uinit) ~= N) || (~isequal(Sz(:,1),SzF(:,1))) || (~all(Sz(:,2)==R))
        error('Wrong Initialization');
    end
elseif strcmp(param.init(1:4),'rand')
    Uinit = cell(N,1);
    for n = 1:N
        Uinit{n} = randn(SzF(n,1),R);
    end
elseif strcmp(param.init(1:4),'nvec')
    % Note: n in inmode: Yn'*Yn = An * ((M*Gammaout*M') .* Gammain_n) * An';
    % Note: n in outmode: Yn'*Yn = An * (Gammaout_n .* (M'*Gammain*M')) * An';
    %Uinit{n} = real(nvecs(X,n,R));
    Jr = param_struct.Jr; cJr = [0;cumsum(Jr(:))];
    dimorder = 1:N;
    inmodes = param_struct.inmodes;
    outmodes = setdiff(dimorder,inmodes);
    
    Rin = numel(Jr);
    Rout = sum(Jr);   %% = SUM(JR)
    Cin = zeros(Rin,Rin,numel(inmodes));
    for k = 1:numel(inmodes)
        n = inmodes(k);
        Cin(:,:,k) = Fact{n}'*Fact{n};
    end
    Cout = zeros(Rout,Rout,numel(outmodes));
    for k = 1:numel(outmodes)
        n = outmodes(k);
        Cout(:,:,k) = Fact{n}'*Fact{n};
    end
    
    
    Gammaout= prod(Cout,3);
    Temp = zeros(Rin,Rout);
    for r = 1:Rin
        idr = cJr(r)+1:cJr(r+1);
        Temp(r,:) = sum(Gammaout(idr,:),1);
    end
    Temp2 = zeros(Rin,Rin);
    for r = 1:Rin
        idr = cJr(r)+1:cJr(r+1);
        Temp2(:,r) = sum(Temp(:,idr),2);
    end
    
    eigsopts = struct('disp',0);
    
    Uinit = cell(N,1);
    for kn = 1:numel(inmodes)
        n = inmodes(kn);
        
        if R < Rin
            [qq,rr] = qr(Fact{n},0);
            
            Gammain_n = prod(Cin(:,:,[1:kn-1 kn+1:end]),3);
            G = rr * (Gammain_n .* Temp2) *rr';
            [u,d] = eigs(G,R, 'LM', eigsopts);
            eigsopts.disp = 0;
            
            Uinit{n} = qq * u;
        elseif  (R == Rin) && (R<=SzF(n))
            [qq,rr] = qr(Fact{n},0);
            Uinit{n} = qq;
        else
            Uinit{n} = (randn(SzF(n),R));
        end
    end
    
    Gammain = prod(Cin,3);
    iJr = ones(1,sum(Jr));
    for r = 1:numel(Jr)
        idr = cJr(r)+1:cJr(r+1);
        iJr(idr) = r;
    end
    Temp = Gammain(iJr,iJr);
    
    for kn = 1:numel(outmodes)
        n = outmodes(kn);
        
%         Gammaout_n = prod(Cout(:,:,[1:kn-1 kn+1:end]),3);
%         Temp2 = Gammaout_n .* Temp;
%         
%         G = Fact{n}' * Temp2 * Fact{n};
%         [u,d] = eigs(G,R, 'LM', eigsopts);
%         eigsopts.disp = 0;
%         
%         Uinit{n} = u;

        if (R < SzF(n)) 
            [qq,rr] = qr(Fact{n},0);
            G = rr *rr';
            [u,d] = eigs(G,R, 'LM', eigsopts);
            eigsopts.disp = 0;
            
            Uinit{n} = qq * u;
        else
            Uinit{n} = (randn(SzF(n),R));
        end
    end
else
    error('Invalid initialization');
end
end

%%

function [Pnew,param_ls] = cp_linesearch(X,P,U0,param_ls)
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