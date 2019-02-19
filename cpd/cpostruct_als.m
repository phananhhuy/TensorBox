function [P,output] = cpostruct_als(Fact,param_struct,R,opts)
% Orthogonallly constrained CPD of structural Kruskal tensors.
% This function is often used with the FCP algorithm.
%
% Input:
%   Fact: factor matrices of a rank-(Lo1) structured tensor Fact: In x Rn
%   param_struct: structure of structured tensor
%       .Jr = 1-D array comprising ranks of block terms
%       Note: Rout = sum(Jr) is rank of Fact(n) where n in outmodes
%           and Rin = numel(Jr) is rank of Fact(n) where n in inmodes        
%   R: rank, R<= Rin.
%   [opts] (Optional)
%    opts.tol: Tolerance for test of convergence. {1e-4}
%    opts.maxiter: Maximum number of iterations. {100}
%    opts.init: Initial guess. [{'nvecs'}|'random'|cell array]
%    opts.orthomodes:  modes which have orthogonal factor matrices
%
% Output:
%  P:  ktensor of estimated factors
%  output:
%      .Fit
%      .NoIters
%
% REF 
% [1] A.-H. Phan, P. Tichavsky, A. Cichocki, "CANDECOMP/PARAFAC
% Decomposition of High-order Tensors Through Tensor Reshaping", arXiv,
% http://arxiv.org/abs/1211.3796, 2012  
%
% Example: see demo_fCP_ortho
%
% See also cpstruct_als, cpo_als1, cpo_als2, cp_fcp
% 
% 
% TENSOR BOX, v1. 2012
% Phan Anh Huy.

 
%% Fill in optional variable
if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    P = param; return
end

%%
N = numel(Fact);
dimorder = 1:N;
inmodes = param_struct.inmodes;
Jr = param_struct.Jr;       % Rin x 2 indicates number of replication of a component
cJr = [0;cumsum(Jr(:))];

outmodes = setdiff(dimorder,inmodes);
nonorthomodes = setdiff(1:N,param.orthomodes);

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

% Ktensor Fact
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
normX = sqrt(Gammain(:)' * ss2(:));
% I = size(X);

%% Initialize factors U
fprintf('\nFast structured CPO_ALS:\n');
%Uinit = cp_init(X,R,param); U = Uinit;
Uinit = cpstruct_init(Fact,param_struct,R,param); U = Uinit;

%% Output
if nargout >=2
    output = struct('Uinit',{Uinit},'NoIters',[]);
    %if param.TraceFit
        output.Fit = [];
    %end
    if param.TraceMSAE
        output.MSAE = [];
    end
end

FtUin = zeros(Rin,R,numel(inmodes));
for k = 1:numel(inmodes)
    n = inmodes(k);
    FtUin(:,:,k) = Fact{n}'*conj(U{n});
end
FtUout = zeros(Rout,R,numel(outmodes));
for k = 1:numel(outmodes)
    n = outmodes(k);
    FtUout(:,:,k) = Fact{n}'*conj(U{n});
end

%P = ktensor(U);
Gammaout = prod(FtUout,3); Gammain = prod(FtUin,3);
ss = zeros(Rin,R);
for r = 1:Rin
    ss(r,:) = sum(Gammaout(cJr(r)+1:cJr(r+1),:),1);
end
innerprd = Gammain(:)' * ss(:);
err =  normX^2 - real(innerprd);
fit = 1-sqrt(err)/normX;
if param.TraceFit
    output.Fit = fit;
end
if param.TraceMSAE
    msae = (pi/2)^2;
end

%% main iteration
for iter = 1:param.maxiters
    fitold = fit;
    
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
        Gammain_n = Gammain_n .* Temp;
        G = Fact{n}*Gammain_n;
        
        if ismember(n,param.orthomodes)    
            [Up, Dp, Vp] = svd(G, 0);
            %sigma = sum(U{n}.*G,1);
            %[Up, Dp, Vp] = svd(G*diag(sigma), 0);
            U{n} = Up*Vp';        
        else
            U{n} = G;
            if n~=nonorthomodes(end)
                U{n} = bsxfun(@rdivide,U{n},sqrt(sum(abs(U{n}).^2)));
            end
        end

        FtUin(:,:,kn) = Fact{n}'*conj(U{n});
    end
        
    Gammain = prod(FtUin,3);
    Gammain = Gammain(repidx,:);
    for kn = 1:numel(outmodes)
        n = outmodes(kn);
        
        Gammaout_n = prod(FtUout(:,:,[1:kn-1 kn+1:end]),3);
        Gammaout_n = Gammaout_n .* Gammain;
        G = Fact{n}*Gammaout_n;
        
        if ismember(n,param.orthomodes)
            %sigma = sum(U{n}.*G,1);
            [Up, Dp, Vp] = svd(G, 0);
            U{n} = Up*Vp';
        else
            U{n} = G;
            if n~=nonorthomodes(end)
                U{n} = bsxfun(@rdivide,U{n},sqrt(sum(abs(U{n}).^2)));    
            end
        end
        FtUout(:,:,kn) = Fact{n}'*conj(U{n});
    end
    
    %P = ktensor(U);
    Gammaout = prod(FtUout,3); Gammain = prod(FtUin,3);
    ss = zeros(Rin,R);
    for r = 1:Rin
        ss(r,:) = sum(Gammaout(cJr(r)+1:cJr(r+1),:),1);
    end
    innerprd = Gammain(:)' * ss(:);
    
    UtU = zeros(R,R,N);
    for n = 1:N
        UtU(:,:,n) = U{n}'*U{n};
    end

    normP2 = sum(sum(prod(UtU,3)));
    normresidual =  sqrt(normX^2 + normP2 - 2 * real(innerprd));
    fit = 1 - (normresidual/ normX); %fraction explained by model
    %if param.TraceFit
    fitchange = abs(fitold - fit);
    stop(1) = fitchange < param.tol;
    stop(3) = fit >= param.fitmax;
    if nargout >=2
        output.Fit = [output.Fit; fit];
    end
    %end
    
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

P = ktensor(U);
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
                fitinit = real(outinit.Fit(end));
                if fitinit > bestfit
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
        elseif  R == Rin
            [qq,rr] = qr(Fact{n},0);
            Uinit{n} = qq;
        else
            Uinit{n} = orth(randn(SzF(n),R));
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

        if R < SzF(n)    
            [qq,rr] = qr(Fact{n},0);
            G = rr *rr';
            [u,d] = eigs(G,R, 'LM', eigsopts);
            eigsopts.disp = 0;
            
            Uinit{n} = qq * u;
        else
            Uinit{n} = orth(randn(SzF(n),R));
        end
    end
else
    error('Invalid initialization');
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
% param.addOptional('linesearch',true);
%param.addParamValue('verify_convergence',true,@islogical);
param.addParamValue('TraceFit',true,@islogical);
param.addParamValue('TraceMSAE',false,@islogical);

param.addOptional('orthomodes',[]);

param.parse(opts);
param = param.Results;
param.verify_convergence = param.TraceFit || param.TraceMSAE;
end
