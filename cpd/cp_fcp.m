function [P,output,BagofOut] = cp_fcp(X,R,opts)
% FCP:  CPD through tensor unFolding
%
% The FCP algorithm decomposes a high order-N tensor X into N factors of
% R components through generalized tensor unfolding.
%
%
% INPUT:
%   X:  order-N tensor which can be a tensor or a ktensor.
%   R:  rank of the approximate tensor
%   OPTS: optional parameters
%     .compress_param   : parameter struct for Tucker compression
%       .compressrawdata: 0/1    compression of the raw tensor X
%       .compress: 1/0    compression of unfolded tensor.
%       .R                (multilinear) rank of the Tucker decomposition,
%                         or size of the compressed tensor. Default value
%                         is rank of CPD.
%       .maxiters         maximum number of iterations for compression
%                         (ussually less than 10)
%       .tol              1e-6
%
%     .foldingrule:
%        -'half' tensor order will be reduced by half at each
%               stage, until the unfolded tensor is of order-3 or "mindim" 
%        -'one'  reduce dimensions by 1
%        -'direct'   force tensor to be order-3.
%        -an array indicates unfoldings,
%               e.g., [2 2 2]: half rule
%                     [2 1 1]: one-rule
%        -an cell array indicates modes to be combined
%               e.g., {1 3 [2 4]}
%
%     .mindim :  order of the unfolded tensor
%
%     .foldingreconstruction_type: 'sequential'
%                   This parameter indicates the reconstruction process for
%                   multi-mode unfolding. 
%                   -'sequential' :  consider multimode-unfolding as
%                   multistages of two-mode unfoldings.
%                   For example, unfolding [1,2,(3,4,5)] can be performed
%                   as two two-modes unfoldings [1,2,(3,(4,5))]
%
% 
%     .cp_func:  CP algorithm to factorize unfolded tensor (@cp_fastals)
%     .cp_param: parameters of the CP algorithm assigned in "cp_func".
%          
%
%     .cp_reffunc: CP algorithm to refine the factor matrices (@cp_fLMa).
%                  This parameter is only used if ".fullrefine" is true.
%     .fullrefine: set it "true" to reestimate the whole factors from the 
%                  data using the solution returned by FCP 
%
%     .cpstruct_func  :  algorithm to approximate a rank-J structured
%                   K-tensor by a rank-R K-tensor. 
%                   This can be "@cpstruct_als", or @cpostruct_als for CPD
%                   with orthogonality constraints.
%                   Any algorithm for structured CPD (PARALIND) can be used
%                   here.
% 
%     .var_thresh   : threshold for truncated SVD. Set this parameter to 0
%                   for rank-one tensor approximation, and higher value for
%                   low-rank approximation.
%
%
% Output:
%  P:  ktensor of estimated factors
%  output:
%  BagofOut: cell array consisting of outputs of estimation process in FCP
%            Use vis_fcp_output to visualize the result.
% REF:
%
% [1] A.-H. Phan, P. Tichavsky, A. Cichocki, "CANDECOMP/PARAFAC
% Decomposition of High-order Tensors Through Tensor Reshaping", arXiv,
% http://arxiv.org/abs/1211.3796, 2012
%
% [2] Petr Tichavsky, Anh Huy Phan, Zbynek Koldovsky, "Cramer-Rao-Induced
% Bounds for CANDECOMP/PARAFAC tensor decomposition", IEEE TSP, in print,
% 2013, available online at http://arxiv.org/abs/1209.3215, .
%
% [3] A. H. Phan, P. Tichavsky, and A. Cichocki, "On fast computation of
% gradients for CP algorithms", http://arxiv.org/abs/1204.1586, 2012,
% submitted.
%
% The function uses the Matlab Tensor toolbox.
% See also: vis_fcp_output, unfoldingstrategy,comparefoldings, foldingrule2char
%           cp_fastals, cp_fLM, fastcribCP, gen_matrix
%
% This algorithm and its Matlab function are a part of the TENSORBOX, 2012.
% Copyright Phan Anh Huy, 04/2012
%
% This software must not be redistributed, or included in commercial
% software distributions without written agreement by its author.
%
% v1. 17.07.2012 fixed lowrank approx. for half-folding rule
% Some bugs when folding rules are not in a right order.

if ~exist('opts','var'), opts = struct; end
param = parseInput(opts);
if nargin == 0
    P = param; return
end
if isfield(opts,'cp_reffunc') && isempty(opts.('cp_reffunc'))
    param.cp_reffunc = param.cp_func;
end
fullrefine = param.fullrefine;
cp_param = param.cp_param;
if fullrefine
    cp_param0 = cp_param;
end

Rfold = param.Rfold;
if isempty(Rfold);
    Rfold = R;
end


% missing tensor
if isa(X,'tensor')
    Weights = tenfun(@isnan,X);Weights = ~logical(Weights.data);
    normX = norm(X(Weights(:)));
    missingdata = any(Weights(:)==0);
    if missingdata == false
        Weights = [];
    else
        Weights = sptensor(Weights);
    end
end
if isa(X,'ktensor')
    Weights = X.weights;
    missingdata = ~(isempty(Weights) || isscalar(Weights) ...
        || (nnz(Weights) == prod(size(Weights))));
    if missingdata == false
        Weights = [];
    end
    normX = norm(X);
end


% if missingdata == true
%     param.compress_param.compress = false;
% end

if param.compress_param.compress == true
    if isempty(param.compress_param.R)
        param.compress_param.R = Rfold;
    end
    if isempty(param.compress_param.maxiters)
        param.compress_param.maxiters = cp_param.maxiters;
    end
    if isempty(param.compress_param.tol)
        param.compress_param.tol = cp_param.tol;
    end
end


if param.TraceFit == true
    cp_param.TraceFit = true;
end
N = ndims(X); In = size(X);
mainfunc = str2func(mfilename);

%% Initialize CP algorithm
switch param.init
    case 'forward'
        U = cp_init(X,Rfold,cp_param);
        U = cellfun(@double,U,'uni',0);
    case 'backward'
        U = cell(N,1);
end

% Reorder dimensions so that I1 <= I2 <= ...<=IN for fast CP gradients
% p_perm = [];
% if ~issorted(In)
%     [In,p_perm] = sort(In);
%     X = permute(X,p_perm);
%     Weights = permute(Weights,p_perm);
%     U = U(p_perm);
% end

fprintf('\nCP_FCP: %s\n',sprintf('%d ',In));

%% Reduction or folding rule
if isnumeric(param.foldingrule)
    if sum(param.foldingrule) == N
        pat1 = param.foldingrule;
    else
        param.foldingrule = 'half';
    end
end
if ischar(param.foldingrule)
    switch param.foldingrule
        case 'direct'
            pat1 = zeros(1,param.mindim);
            cpI = cumprod(In(end:-1:1));
            %pat1(1) = find(cpI <= cpI(end)^(1/param.mindim),1,'last');
            [~,ndimpat1] = min(abs(cpI - cpI(end)^(1/param.mindim)));
            pat1(1) = ndimpat1;
            for n = 2:param.mindim-1
                cpI = cumprod(In(end:-1:pat1(n-1)+1));
                [~,ndimpat1] = min(abs(cpI - cpI(end)^(1/(param.mindim-n+1))));
                pat1(n) = ndimpat1;
                %pat1(n) = find(cpI < cpI(end)^(1/(param.mindim-n+1)),1,'last');
            end
            pat1(param.mindim) = N-sum(pat1(1:end-1));
            pat1 = pat1(end:-1:1);
            param.fullrefine = false;
            
            
        case 'half'
            if N > 4
                pat1 = 2*ones(1,floor(N/2));
                if mod(N,2) == 1
                    pat1 = [pat1 1];
                end
            elseif N==4
                pat1 = [2 1 1];
            end
            param.fullrefine = false;
            
        case 'one'
            pat1 = [2 ones(1,N-2)];
            param.fullrefine = false;
    end
    
    order = 1:N;
    orderidx = mat2cell(order,1,pat1);
    
elseif iscell(param.foldingrule)
    % Permute tensor so that folding modes in the ascending order
    orderidx2 = cellfun(@(x) sort(x,'ascend'),param.foldingrule,'uni',0);
    order2 = cell2mat(orderidx2);
    pat1 = cellfun(@numel,orderidx2);
    
    % change "orderidx" to retranpose the output tensor
    order = 1:N;
    orderidx = mat2cell(1:N,1,pat1);
    
    if ~issorted(order2)
        X = permute(X,order2);
        param.foldingrule = pat1;
        param.mindim = numel(param.foldingrule);
        In = In(order2);
        U = U(order2);
    end
    
    %orderidx = cellfun(@(x) sort(x,'ascend'),param.foldingrule,'uni',0);
    %order = cell2mat(orderidx);
    %pat1 = cellfun(@numel,orderidx);
else
    order = 1:N;
    orderidx = mat2cell(order,1,pat1);
end

%% Main loop

output = struct('Exectime',struct('compress',0,'cp',0,'r1aprox',0,'ref',0,'other',0),...
    'Fit',[],'CP_output',[],'Sublayer',[]);
outputlayer = cell(1,param.refine);
BagofOut = cell(1);

Nnew= numel(pat1);
Uinit = cell(Nnew,1);


%%

% Compress the raw tensor
compressedrawdata = false;
if param.compress_param.compressrawdata
    dimorder = find(In>param.compress_param.R);
    if ~isempty(dimorder)
        ts = tic;
        %Xn = tucker_als(Xn,min(size(Xn),[R R R]));
        [Xcpr,fitcompress] = mtucker_als(X,param.compress_param.R(ones(1,numel(dimorder))),...
            struct('dimorder',dimorder,'maxiters',param.compress_param.maxiters,...
            'tol',param.compress_param.tol,'printitn',cp_param.printitn));
        tcmpress = toc(ts);
        output.Exectime.compress = output.Exectime.compress + tcmpress;
        BagofOut{end+1} = struct('Name','Compress','Time', tcmpress,'Fit',fitcompress,'Level',1);
        
        compressedrawdata = true;
        %Xraw = X;
        if isa(Xcpr,'ttensor')
            X = Xcpr.core;
        elseif isa(Xcpr,'ktensor')
            X = Xcpr.lambda;
        end
        In = size(X);
    end
    param.compress_param.compressrawdata = false;
end
%%
%eigsopts.disp = 0;
% n = 0;
k = 1;
% for k = 1:max(1,param.refine)
%     n = n+1;
n = 1;
% order = [n:N 1:n-1];
% orderidx = mat2cell(order,1,pat1);

% Tensor unfolding
ts = tic;
Inew = cellfun(@(x) prod(In(x)),orderidx);
clear Xn;
if isa(X,'tensor') || isa(X,'sptensor')
    if ~all(order == 1:N)
        Xn = permute(X,order);Xn = reshape(Xn,Inew); % folding
    else
        Xn = reshape(X,Inew);
    end
elseif isa(X,'ktensor')
    Ux = cell(Nnew,1);
    for n = 1:numel(orderidx)
        Ux{n} = khatrirao(X.U(fliplr(orderidx{n})));
    end
    if ~isscalar(Weights) && ~isempty(Weights)
        Weights2 = permute(Weights,order);Weights2 = reshape(Weights2,Inew); % folding
        Xn = ktensor(X.lambda,Ux,Weights2);
    else
        Xn = ktensor(X.lambda,Ux);
    end
end
Xnbk = Xn;
%     % missing tensor
%     if isa(Xn,'tensor')
%         Weights = tenfun(@isnan,Xn);
%         missingdata = any(Weights.data);
%         if missingdata == false
%             Weights = [];
%         else
%             Weights = sptensor(~Weights);
%         end
%     end
%     if isa(Xn,'ktensor')
%         Weights = Xn.weights;
%         missingdata = isempty(Weights) || isscalar(Weights) ...
%             || (nnz(Weights) == prod(size(Weights)));
%         if missingdata == false
%             Weights = [];
%         end
%     end
t1 = toc(ts);
output.Exectime.other = output.Exectime.other + t1;

% Compress the unfolded tensor
if param.compress_param.compress
    
    dimorder = find(Inew>param.compress_param.R);
    if ~isempty(dimorder)
        ts = tic;
        %Xn = tucker_als(Xn,min(size(Xn),[R R R]));
        [Xn,fitcompress] = mtucker_als(Xn,param.compress_param.R(ones(1,numel(dimorder))),...
            struct('dimorder',dimorder,'maxiters',param.compress_param.maxiters,...
            'tol',param.compress_param.tol,'printitn',cp_param.printitn));
        t2 = toc(ts);
        output.Exectime.compress = output.Exectime.compress + t2;
        BagofOut{end+1} = struct('Name','Compress','Time', t1 + t2,'Fit',fitcompress,'Level',1);
    end
end


% CP initialize
switch param.init
    case 'forward'
        ts = tic;
        for nnew = 1:Nnew
            Uinit{nnew} = khatrirao(U(fliplr(orderidx{nnew})));
        end
        t1 = toc(ts);
        
        if isa(Xn,'ttensor')
            Uinit = cellfun(@(x,y) x'*y,Xn.U,Uinit,'uni',0);
        end
        cp_param.init = Uinit; %cp_param.init ={Uinit 'nvecs'}; Uinit;%'random';
        
        output.Exectime.other = output.Exectime.other + t1;
end

% CP factorization of the folding tensor Xn
if isa(Xn,'tensor') || isa(Xn,'ktensor') || isa(Xn,'sptensor')
    cp_param.alsinit = 0;
    param.cp_param = cp_param;
    if ndims(Xn)>param.mindim
        param.recurlevel = param.recurlevel+1;
        [T,output2,BagofOut2] = mainfunc(Xn,Rfold,param);
        outputlayer{k} = output2;
        param.recurlevel = param.recurlevel-1;
        % Update level of BagofOut
        for kbag = 1:numel(BagofOut2)
            BagofOut2{kbag}.Level = BagofOut2{kbag}.Level + 1;
        end
        BagofOut(end+1:end+numel(BagofOut2)) = BagofOut2;
        %output = updateoutput(output,output2);
        
    else
        %cp_param.tol = 1e-4;
        ts = tic;
        [T,outputcp] = cp_eval(Xn,Rfold,param.cp_func,cp_param);
        t3 = toc(ts);
        output.Exectime.cp= output.Exectime.cp + t3;
        output.CP_output = outputcp;
        output.Fit = outputcp.Fit;
        
        BagofOut{end+1} = struct('Name','CP','Time',t3,'Fit',outputcp.Fit,'Level',1);
        
    end
elseif isa(Xn,'ttensor')
    cp_param.alsinit = 0;
    param.cp_param = cp_param;
    if ndims(Xn)>param.mindim
        param.recurlevel = param.recurlevel+1;
        [T,output2,BagofOut2] = mainfunc(Xn.core,Rfold,param);
        param.recurlevel = param.recurlevel-1;
        %exectime = updatetime(exectime,output2.exectime);
        T = ttm(T,Xn.U);
        %output = updateoutput(output,output2);
        outputlayer{k} = output2;
        
        % Update level of BagofOut
        for kbag = 1:numel(BagofOut2)
            BagofOut2{kbag}.Level = BagofOut2{kbag}.Level + 1;
        end
        BagofOut(end+1:end+numel(BagofOut2)) = BagofOut2;
    else
        %cp_param.tol = 1e-4;
        ts = tic;
        [T,outputcp] = cp_eval(Xn.core,Rfold,param.cp_func,cp_param);
        T = ttm(T,Xn.U);
        t3 = toc(ts);
        output.Exectime.cp = output.Exectime.cp + t3;
        output.CP_output = outputcp;
        output.Fit = outputcp.Fit;
        
        BagofOut{end+1} = struct('Name','CP','Time',t3,'Fit',outputcp.Fit,'Level',1);
        
    end
    
    %% Correct fit over compression
    if isa(Xn,'ttensor')
        normXn = norm(Xn.core);
    else
        normXn = norm(Xn);
    end
    errorcompress = (1- fitcompress(end))*normX;
    Fit = BagofOut{end}.Fit;
    error = (1 - Fit(:,2)) * normXn;
    error = error.^2 + errorcompress^2;
    Fit(:,2) = 1 - sqrt(error)/normX;
    BagofOut{end}.Fit = real(Fit);
end
cp_param.tol = param.cp_param.tol;
%%
rho = ones(Rfold,1);
%     Cr = ones(Rfold,Rfold);
%     for n = 1:ndims(T)
%         Cr = Cr.* (T.U{n}'*T.U{n});
%     end
%     Cr = (T.lambda'*T.lambda).*Cr;
%
%     for r = 1:Rfold
%         %ur = cellfun(@(x) x(:,r),T.U,'uni',0);
%         %rho(r) =  sum(Cr(r,:)) + sum(Cr(:,r)) - Cr(r,r) -  2 *ttv(Xn,ur);
%         rho(r) =  sum(Cr(r,:)) + sum(Cr(:,r)) - Cr(r,r);
%     end
%     rho = 1-rho/sum(Cr(:)) ;
%     rho = sqrt(rho/max(rho));
%%
% Xn = permute(X,order);Xn = reshape(Xn,Inew); % unfolding
% 1- sqrt(norm(Xn)^2 + norm(T)^2 - 2 * innerprod(Xn,T))/norm(Xn)


ts = tic;
T.U{1} = bsxfun(@times,T.U{1},T.lambda(:).');
% MAP T.U to CP factors
pat2 = cellfun(@numel,orderidx);
nonfolddim = find(pat2 == 1);
for nnew = nonfolddim
    idnnew = orderidx{nnew};
    U{idnnew} = T.U{nnew};
end
folddim = setdiff(1:Nnew,nonfolddim);
for nnew = folddim
    idnnew = orderidx{nnew};
    U{idnnew(1)} = T.U{nnew};
    for knnw = 2:numel(idnnew)
        U{idnnew(knnw)} = [];
    end
end

%     for nnew = folddim
%
%         idnnew = orderidx{nnew};
%         Innew = In(idnnew);
%
%         Jrr = zeros(Rfold,1);M = [];
%         Unnew = U{idnnew(1)};
%         U{idnnew(1)} = [];
%         Rfold = size(Unnew,2);
%         rho = ones(Rfold,1);
%         for r = 1:Rfold
%             V = Unnew(:,r);
%             V = reshape(V,Innew);
%             normV = norm(V(:))^2 ;
%             if numel(idnnew) == 2
%                 % Fold two-modes, approximate a folding component by
%                 % truncated SVD
%
%                 eigopts = struct('disp',0);
%                 sv = svd(V);
%                 varexpl = cumsum(sv.^2);varexpl = varexpl/normV;
%                 Jr = find(varexpl >= param.var_thresh * max(0.95,rho(r)),1,'first');
%                 %Jr = 3;
%                 %if isempty(Jr)
%                 %    1
%                 %end
%                 % low rank approximation
%                 [uv,sv,vv] = svds(V,Jr,'L',eigopts);
%                 sv = diag(sv);
%                 %varexpl = cumsum(sv.^2);varexpl = varexpl/varexpl(end);
%                 %Jr = find(varexpl >= param.var_thresh,1,'first');
%                 uv = uv(:,1:Jr);
%                 vv = vv(:,1:Jr) * diag(sv(1:Jr));
%
%
%                 %                 [uv,sv,vv] = svds(V,2,'L',eigopts);
%                 %                 vv = vv * sv;sv = diag(sv);
%                 %                 varexpl = cumsum(sv.^2);varexpl = varexpl/normV;
%                 %                 Jr = find(varexpl >= param.var_thresh,1,'first');
%                 %                 if isempty(Jr)
%                 %                     % low rank approximation
%                 %                     [uv,sv,vv] = svd(V);sv = diag(sv);
%                 %                     varexpl = cumsum(sv.^2);varexpl = varexpl/varexpl(end);
%                 %                     Jr = find(varexpl >= param.var_thresh,1,'first');
%                 %                     uv = uv(:,1:Jr);
%                 %                     vv = vv(:,1:Jr) * diag(sv(1:Jr));
%                 %                 end
%
%
%                 U{idnnew(1)}(:,end+1:end+Jr) = uv(:,1:Jr);
%                 U{idnnew(2)}(:,end+1:end+Jr) = vv(:,1:Jr);
%                 %M = blkdiag(M,ones(1,Jr));
%
%             else % fold more than 2 modes
%
%                 %if strcmp(param.multimodereconstruction,'direct')
%                 %                 % Tucker reconstruction
%                 %                 V = tensor(V);
%                 %                 r1param = struct('maxiters',1000,'tol',1e-6,'printitn',0);
%                 %                 Tr = mtucker_als(V,ones(1,ndims(V)),r1param);
%                 %                 %Tr = cp_fLMa_v2(V,1,r1param);
%                 %                 varexpl = 1-sqrt(1-Tr.lambda^2/normV); % fit instead of varexpl
%                 %                 if varexpl < param.var_thresh
%                 %                     Jv = zeros(1,ndims(V));
%                 %                     for nv = 1:ndims(V)
%                 %                         Vn = double(tenmat(V,nv));
%                 %                         Vn = Vn * Vn';
%                 %                         sv = eig(Vn);sv = sort(sv,'descend');
%                 %                         %varexpl = cumsum(sv);varexpl = varexpl/normV;
%                 %                         varexpl = 1-real(sqrt(1-cumsum(sv)/normV)); % fit instead of varexpl
%                 %                         Jv(nv) = find(varexpl >= param.var_thresh,1,'first');
%                 %                     end
%                 %                     Tr = mtucker_als(V,Jv,r1param);
%                 %                     %Tr = cp_eval(V,max(Jv),param.cp_func,r1param);
%                 %                     if isa(Tr,'ttensor')
%                 %                         sv = double(Tr.core(:));
%                 %                     elseif isa(Tr,'ktensor')
%                 %                         sv = T.lambda;
%                 %                     end
%                 %                     [~,svix] = sort(abs(sv),'descend');sv = sv(svix);
%                 %                     %varexpl = cumsum(sv.^2);varexpl = varexpl/normV;
%                 %                     varexpl = 1-sqrt(1-cumsum(sv.^2)/normV); % fit instead of varexpl
%                 %                     Jr = find(varexpl >= param.var_thresh,1,'first');
%                 %                     if isempty(Jr)
%                 %                         Jr = numel(varexpl);
%                 %                     end
%                 %                     svix = svix(1:Jr);
%                 %                     svix = ind2sub_full(Jv,svix);
%                 %                     Uv = cell(size(svix,2),1);
%                 %                     for kix = 1:size(svix,2)
%                 %                         Uv{kix} = Tr.U{kix}(:,svix(:,kix));
%                 %                     end
%                 %                     Uv{end} = Uv{end} * diag(sv(1:Jr));
%                 %                 else
%                 %                     Uv = Tr.u; Uv{end} = Uv{end} * diag(Tr.lambda);
%                 %                     Jr = 1;
%                 %                 end
%                 %                 for kn = 1:numel(idnnew)
%                 %                     U{idnnew(kn)}(:,end+1:end+Jr) = Uv{kn};
%                 %                 end
%
%                 % CP reconstruction
%                 V = tensor(V);
%                 r1param = struct('maxiters',1000,'tol',1e-6,'printitn',0);
%                 Tr = mtucker_als(V,ones(1,ndims(V)),r1param);
%                 %Tr = cp_fLMa_v2(V,1,r1param);
%                 varexpl = 1-sqrt(1-Tr.lambda^2/normV); % fit instead of varexpl
%                 if varexpl < param.var_thresh
%                     Jv = zeros(1,ndims(V));
%                     for nv = 1:ndims(V)
%                         Vn = double(tenmat(V,nv));
%                         Vn = Vn * Vn';
%                         sv = eig(Vn);sv = sort(sv,'descend');
%                         %varexpl = cumsum(sv);varexpl = varexpl/normV;
%                         varexpl = 1-real(sqrt(1-cumsum(sv)/normV)); % fit instead of varexpl
%                         Jv(nv) = find(varexpl >= param.var_thresh,1,'first');
%                     end
%                     %Tr = mtucker_als(V,Jv,r1param);
%                     Tr = cp_eval(V,max(Jv),param.cp_func,r1param);
%                     if isa(Tr,'ttensor')
%                         sv = double(Tr.core(:));
%                     elseif isa(Tr,'ktensor')
%                         sv = T.lambda;
%                     end
%                     [~,svix] = sort(abs(sv),'descend');sv = sv(svix);
%                     %varexpl = cumsum(sv.^2);varexpl = varexpl/normV;
%                     varexpl = 1-sqrt(1-cumsum(sv.^2)/normV); % fit instead of varexpl
%                     Jr = find(varexpl >= param.var_thresh,1,'first');
%                     if isempty(Jr)
%                         Jr = numel(varexpl);
%                     end
%                     svix = svix(1:Jr);
%                     svix = ind2sub_full(Jv,svix);
%                     Uv = cell(size(svix,2),1);
%                     for kix = 1:size(svix,2)
%                         Uv{kix} = Tr.U{kix}(:,svix(:,kix));
%                     end
%                     Uv{end} = Uv{end} * diag(sv(1:Jr));
%                 else
%                     Uv = Tr.u; Uv{end} = Uv{end} * diag(Tr.lambda);
%                     Jr = 1;
%                 end
%                 for kn = 1:numel(idnnew)
%                     U{idnnew(kn)}(:,end+1:end+Jr) = Uv{kn};
%                 end
%
%                 %                 elseif strcmp(param.multimodereconstruction,'sequential')
%                 %
%                 %                 end
%
%                 %M = blkdiag(M,ones(1,Jr));
%             end
%             Jrr(r) = Jr;
%         end
%
%         if param.recurlevel > 0;
%             if sum(Jrr) > Rfold
%
%                 fprintf('Rank-%d to Rank-%d CPD\n',sum(Jrr),Rfold);
%
%                 actdim = find(~cellfun(@isempty,U));
%
%                 param_struct.Jr = Jrr;
%                 [diffdim,dimid] = setdiff(actdim,idnnew);
%                 param_struct.inmodes = dimid;
%
%
%                 % Estimate rank-R CPD from rank-(J1+J2+..+JR) CP tensor
%                 % Expand factors
%                 %actdim = find(~cellfun(@isempty,U));
%                 %                 for n1 = setdiff(actdim,idnnew)
%                 %                     U{n1} = U{n1} * M;
%                 %                 end
%                 UJ = U(actdim);UR = cell(numel(actdim),1);
%                 cJr = [0;cumsum(Jrr(:))];
%                 for n1 = 1:numel(diffdim)
%                     UR{dimid(n1)} = U{diffdim(n1)}(:,1:Rfold);
%                 end
%                 for n1 = idnnew
%                     UR{n1} = U{n1}(:,cJr(1:Rfold)+1);
%                 end
%
%                 cp_param.init = UR;
%                 cp_param.tol = min(1e-8,cp_param.tol);
%                 %PR = cp_eval(PJ,Rfold,param.cp_reffunc,cp_param);
%                 PR = cpstruct_als(UJ,param_struct,Rfold,cp_param);
%                 PR = arrange(PR);
%                 U(actdim) = PR.U;
%                 U{actdim(1)} = U{actdim(1)} * diag(PR.lambda);
%
%                 %             P = reshape(double(full(X)),size(PR));P = tensor(P);
%                 %             fitr1 = 1- real(sqrt(normX^2 + norm(PJ)^2 - 2 * innerprod(P,PJ)))/normX
%                 %             fitr1 = 1- real(sqrt(normX^2 + norm(PR)^2 - 2 * innerprod(P,PR)))/normX
%             end
%         else
%             if (sum(Jrr) > Rfold) && (nnew ~= folddim(end))
%                 %                 Rfold2 = min(Rfold+10,sum(Jrr));
%                 %                 fprintf('Rank-%d to Rank-%d CPD\n',sum(Jrr),Rfold);
%                 %
%                 %                 % Estimate rank-R CPD from rank-(J1+J2+..+JR) CP tensor
%                 %                 % Expand factors
%                 %                 actdim = find(~cellfun(@isempty,U));
%                 %                 param_struct.M = M;
%                 %                 param_struct.Jr = Jrr;
%                 %                 [diffdim,dimid] = setdiff(actdim,idnnew);
%                 %                 param_struct.inmodes = dimid;
%                 %
%                 %
%                 %                 UJ = U(actdim);
%                 %                 cp_paramstruct = cp_param;
%                 %                 cp_paramstruct.tol = min(1e-6,cp_param.tol);
%                 %                 cp_paramstruct.TraceMSAE = false;
%                 %                 cp_paramstruct.init = 'random';
%                 %                 PR = cpstruct_als(UJ,param_struct,Rfold2,cp_paramstruct); % need linesearch of fLM
%                 %                 PR = arrange(PR);
%                 %                 U(actdim) = PR.U;
%                 %                 U{actdim(1)} = U{actdim(1)} * diag(PR.lambda);
%
%                 %
%                 fprintf('Rank-%d to Rank-%d CPD\n',sum(Jrr),Rfold);
%
%                 % Estimate rank-R CPD from rank-(J1+J2+..+JR) CP tensor
%                 % Expand factors
%                 actdim = find(~cellfun(@isempty,U));
%                 param_struct.M = M;
%                 param_struct.Jr = Jrr;
%                 [diffdim,dimid] = setdiff(actdim,idnnew);
%                 param_struct.inmodes = dimid;
%
%
%                 UJ = U(actdim);
%                 cJr = [0;cumsum(Jrr(:))];
%                 UR = cell(numel(actdim),1);
%                 for n1 = 1:numel(diffdim)
%                     UR{dimid(n1)} = U{diffdim(n1)}(:,1:Rfold);
%                 end
%                 for n1 = idnnew
%                     UR{n1} = U{n1}(:,cJr(1:Rfold)+1);
%                 end
%
%                 cp_paramstruct = cp_param;
%                 cp_paramstruct.init = UR;
%                 cp_paramstruct.tol = min(1e-8,cp_param.tol);
%                 cp_paramstruct.TraceMSAE = false;
%                 PR = cpstruct_als(UJ,param_struct,Rfold,cp_paramstruct);
%
%                 %                 PJ = ktensor(UJ,Weights); PJ = arrange(PJ);
%                 %                 PR = ktensor(PJ.lambda(1:R),cellfun(@(x) x(:,1:R),PJ.u,'uni',0));
%                 %                 cp_param.init = PR;
%                 %PR = cp_eval(PJ,R,param.cp_reffunc,cp_param);
%                 PR = arrange(PR);
%                 U(actdim) = PR.U;
%                 U{actdim(1)} = U{actdim(1)} * diag(PR.lambda);
%
%             elseif (sum(Jrr) > R)
%                 fprintf('Rank-%d to Rank-%d CPD\n',sum(Jrr),R);
%
%                 % Estimate rank-R CPD from rank-(J1+J2+..+JR) CP tensor
%                 % Expand factors
%                 actdim = find(~cellfun(@isempty,U));
%                 param_struct.M = M;
%                 param_struct.Jr = Jrr;
%                 [diffdim,dimid] = setdiff(actdim,idnnew);
%                 param_struct.inmodes = dimid;
%
%
%                 UJ = U(actdim);UR = cell(numel(actdim),1);
%                 cJr = [0;cumsum(Jrr(:))];
%                 for n1 = 1:numel(diffdim)
%                     UR{dimid(n1)} = U{diffdim(n1)}(:,1:R);
%                 end
%                 for n1 = idnnew
%                     UR{n1} = U{n1}(:,cJr(1:R)+1);
%                 end
%
%                 cp_paramstruct = cp_param;
%                 cp_paramstruct.init = UR;
%                 cp_paramstruct.tol = min(1e-8,cp_param.tol);
%                 cp_paramstruct.TraceMSAE = false;
%                 PR = cpstruct_als(UJ,param_struct,R,cp_paramstruct);
%
%                 %                 PJ = ktensor(UJ,Weights); PJ = arrange(PJ);
%                 %                 PR = ktensor(PJ.lambda(1:R),cellfun(@(x) x(:,1:R),PJ.u,'uni',0));
%                 %                 cp_param.init = PR;
%                 %PR = cp_eval(PJ,R,param.cp_reffunc,cp_param);
%                 PR = arrange(PR);
%                 U(actdim) = PR.U;
%                 U{actdim(1)} = U{actdim(1)} * diag(PR.lambda);
%             end
%         end
%     end


[U,blkname] = cp_2mode_reconstruction(U,orderidx,Nnew,nonfolddim,param,In,R,Rfold,cp_param);

t4 = toc(ts);
output.Exectime.r1aprox = output.Exectime.r1aprox + t4;

if ~isscalar(Weights) && ~isempty(Weights)
    P = ktensor(U(:),Weights);
else
    P = ktensor(U(:));
end
fitr1 = 1- real(sqrt(normX^2 + norm(P)^2 - 2 * real(innerprod(X,P))))/normX;
%if sum(Jrr) > R
%blkname = 'lowrank';
%else
%    blkname = 'Rank1';
%end
BagofOut{end+1} = struct('Name',blkname,'Time',t4,'Fit',fitr1,'Level',1);
% end
% 1- sqrt(norm(X)^2 + norm(P)^2 - 2 * innerprod(X,P))/norm(X)
%%
Rfold = size(U{1},2);
if R ~= Rfold
    fprintf('Rank-%d to Rank-%d CPD\n',Rfold,R);
    
    if R<=Rfold
        PR = ktensor(P.lambda(1:R),cellfun(@(x) x(:,1:R),P.u,'uni',0),Weights);
        cp_param.init = PR;
    else
        cp_param.init = 'nvecs';
    end
    ts = tic;
    [P,outputcp] = cp_eval(P,R,param.cp_func,cp_param);
    t5 = toc(ts);
    P = arrange(P);
    
    fit = 1- real(sqrt(normX^2 + norm(P)^2 - 2 * innerprod(X,P)))/normX;
    
    %exectime.layer = exectime;
    output.Exectime.ref = t5;
    output.CPred_output = outputcp;
    output.Fit = fit;
    
    BagofOut{end+1} = struct('Name','CP_Rfold->R','Time',t5,'Fit',fit,'Level',1);
end

if fullrefine
    fprintf('Refining stage: Rank-%d CPD\n',R);
    
    cp_param0.alsinit = 0;
    cp_param0.init = P;%'random';
    ts = tic;
    [P,outputcp] = cp_eval(X,R,param.cp_reffunc,cp_param0);
    t6 = toc(ts);
    P = arrange(P);
    %exectime.layer = exectime;
    output.Exectime.ref = t6;
    output.CPref_output = outputcp;
    output.Fitref = outputcp.Fit;
    
    BagofOut{end+1} = struct('Name','CPRef','Time',t6,'Fit',outputcp.Fit,'Level',1);
end

%% Back projection if compressing the raw data
if compressedrawdata == true
    P = ttm(P,Xcpr.U);
end
% Permute the results
if ~issorted(order) % this tranposition may not be needed
    P = ipermute(P,order);
end
if exist('order2','var') && ~issorted(order2)
    P = ipermute(P,order2);
end
%%
output.Sublayer = outputlayer;
BagofOut(cellfun(@isempty,BagofOut)) = [];
%% Arrange the final tensor so that the columns are normalized.
% Rearrange dimension of the estimation tensor
% if ~isempty(p_perm)
%     P = ipermute(P,p_perm);
% end

% Check collinearity of the estimated factor matrices and suggest a further
% decomposition if needed

suggestedfoldingrule = unfoldingstrategy(P.U,numel(pat1));
decision = comparefoldings(orderidx,suggestedfoldingrule);
if decision == false
    warning('A further decomposition is needed to improve the performance.')
    warning('Suggested unfolding rule  %s\n',foldingrule2char(suggestedfoldingrule));
else
    fprintf('The unfolding rule %s is optimal.\n', foldingrule2char(orderidx))
end

%%
fprintf('Finish.\n');




    function [T,output] = cp_eval(Xn,R,cp_func,cp_param)
        switch lower(func2str(cp_func))
            case 'cp_opt'
                [T,~,output] = cp_func(Xn,R,'alg','ncg','init',...
                    cp_param.init,'alg_options',cp_param);
                
            case {'swatld' 'dgn' 'pmf3'}
                if N == 3
                    % used for the PARAFAC3W toolkit
                    cpopts = ParOptions;
                    cpopts.algorithm = lower(func2str(cp_func));
                    cpopts.compress = 'off';
                    cpopts.display = 'none';
                    cpopts.convcrit.maxiter = cp_param.maxiters;
                    cpopts.convcrit.relfit = cp_param.tol;
                    [A,B,C,output] = Parafac3W_mdtime(double(Xn),R,cpopts,...
                        cp_param.init{1},cp_param.init{2},cp_param.init{3});
                    T = ktensor({A,B,C});
                else
                    T = ktensor(cp_param.init);
                    output = [];
                end
                
            case {'als' 'parafac'}
                if isa(Xn,'tensor')
                    Weights2 = ~tenfun(@isnan,Xn);
                    Weights2 = Weights2.data;
                elseif isa(Xn,'ktensor')
                    Weights2 = logical(double(Xn.weights));
                    %if isa(Weights2,'tensor'), Weights2 = Weights2.data; end
                    Xn = full(Xn,true);
                end
                
                %OptionsALS = [cp_param.tol 0 0 0 nan cp_param.maxiters];
                OptionsALS = [cp_param.tol 0 0 0 nan cp_param.maxiters];
                
                if ~isfield(cp_param,'init') || isempty(cp_param.init) %|| isempty(cp_param.init{1})
                    [U,it,err]  = parafac(double(Xn),R,OptionsALS,[],0,[]);
                elseif iscell(cp_param.init) && ischar(cp_param.init{1})
                    [U,it,err]  = parafac(double(Xn),R,OptionsALS,[],10,[]);
                else
                    [U,it,err]  = parafac(double(Xn),R,OptionsALS,[],cp_param.init,[]);
                end
                
                T = ktensor(U);
                %Tf = full(T);
                %output.Fit = 1-norm(Xn(Weights2(:)) - Tf(Weights2(:)))/norm(Xn(Weights2(:)));
                output.Fit = 1-sqrt(err)/norm(Xn(Weights2(:)));
            otherwise
                [T,output] = cp_func(Xn,R,cp_param);
        end
        if ~isfield(output,'Fit')
            output.Fit = [];
        end
        
        if (nargout >=2)
            if isempty(output.Fit);
                normX = norm(Xn);
                err = normX^2 + norm(T)^2 - 2 * innerprod(Xn,T);
                output.Fit = [nan 1-sqrt(err)/normX];
            elseif size(output.Fit,2) == 1
                output.Fit = [(1:numel(output.Fit))' output.Fit(:)];
            end
        end
    end





%%  function to approximate folding components by sequential low-rank SVD

    function [U,recontructtype] = cp_2mode_reconstruction(U,orderidx,Nnew,nonfolddim,param,...
            In,R,Rfold,cp_param)
        
        folddimold = folddim;
        folddim = setdiff(1:Nnew,nonfolddim);
        recontructtype = 'rank1'; % for BagOfOutput's name
        % loc = cellfun(@isempty,U);
        % PR = ktensor(U(~loc));
        % P = reshape(double(full(X)),size(PR));P = tensor(P);
        % fitlr = 1- real(sqrt(normX^2 + norm(PR)^2 - 2 * innerprod(P,PR)))/normX;
        % disp(fitlr)
        
        for nnew = folddim
            
            idnnew = orderidx{nnew};
            Innew = In(idnnew);
            
            
            if numel(idnnew) == 2 % Matrix
                tsrec = tic;
                Jrr = zeros(Rfold,1);M = [];
                Unnew = U{idnnew(1)};
                U{idnnew(1)} = []; % reconstructed later.
                Rfold = size(Unnew,2);
                rho = ones(Rfold,1);
                for r = 1:Rfold
                    V = Unnew(:,r);
                    V = reshape(V,Innew);
                    normV = norm(V(:))^2 ;
                    if isa(V,'single') , V = double(V);end
                    
                    % Fold two-modes, approximate a folding component by
                    % truncated SVD
                    
                    eigopts = struct('disp',0);
                    sv = svd(V);  %hold on; plot(sv/sv(1),'m')
                    varexpl = cumsum(sv.^2);varexpl = varexpl/normV;
                    Jr = find(varexpl >= param.var_thresh * max(0.95,rho(r)),1,'first');
                    % low rank approximation
                    [uv,sv,vv] = svds(V,Jr,'L',eigopts);
                    sv = diag(sv);
                    %varexpl = cumsum(sv.^2);varexpl = varexpl/varexpl(end);
                    %Jr = find(varexpl >= param.var_thresh,1,'first');
                    uv = uv(:,1:Jr);
                    vv = conj(vv(:,1:Jr)) * diag(sv(1:Jr));
                    
                    
                    %                 [uv,sv,vv] = svds(V,2,'L',eigopts);
                    %                 vv = vv * sv;sv = diag(sv);
                    %                 varexpl = cumsum(sv.^2);varexpl = varexpl/normV;
                    %                 Jr = find(varexpl >= param.var_thresh,1,'first');
                    %                 if isempty(Jr)
                    %                     % low rank approximation
                    %                     [uv,sv,vv] = svd(V);sv = diag(sv);
                    %                     varexpl = cumsum(sv.^2);varexpl = varexpl/varexpl(end);
                    %                     Jr = find(varexpl >= param.var_thresh,1,'first');
                    %                     uv = uv(:,1:Jr);
                    %                     vv = vv(:,1:Jr) * diag(sv(1:Jr));
                    %                 end
                    
                    
                    U{idnnew(1)}(:,end+1:end+Jr) = uv(:,1:Jr);
                    U{idnnew(2)}(:,end+1:end+Jr) = vv(:,1:Jr);
                    %M = blkdiag(M,ones(1,Jr));
                    
                    Jrr(r) = Jr;
                    
                end
                
                
                if param.recurlevel > 0;
                    if sum(Jrr) > Rfold
                        recontructtype = 'lowrank';
                        
                        fprintf('Rank-%d to Rank-%d CPD\n',sum(Jrr),Rfold);
                        
                        actdim = find(~cellfun(@isempty,U));
                        
                        param_struct.Jr = Jrr;
                        [diffdim,dimid] = setdiff(actdim,idnnew);
                        param_struct.inmodes = dimid;
                        
                        
                        % Estimate rank-R CPD from rank-(J1+J2+..+JR) CP tensor
                        % Expand factors
                        %actdim = find(~cellfun(@isempty,U));
                        %                 for n1 = setdiff(actdim,idnnew)
                        %                     U{n1} = U{n1} * M;
                        %                 end
                        UJ = U(actdim);UR = cell(numel(actdim),1);
                        cJr = [0;cumsum(Jrr(:))];
                        for n1 = 1:numel(diffdim)
                            UR{dimid(n1)} = U{diffdim(n1)}(:,1:Rfold);
                        end
                        for n1 = idnnew
                            UR{n1} = U{n1}(:,cJr(1:Rfold)+1);
                        end
                        
                        cp_param.init = UR;
                        cp_param.tol = min(1e-8,cp_param.tol);
                        %PR = cp_eval(PJ,Rfold,param.cp_reffunc,cp_param);
                        cp_paramstruct.printitn = 0;
                        PR = param.cpstruct_func(UJ,param_struct,Rfold,cp_param);
                        PR = arrange(PR);
                        U(actdim) = PR.U;
                        U{actdim(1)} = U{actdim(1)} * diag(PR.lambda);
                        
                        %             P = reshape(double(full(X)),size(PR));P = tensor(P);
                        %             fitr1 = 1- real(sqrt(normX^2 + norm(PJ)^2 - 2 * innerprod(P,PJ)))/normX
                        %             fitr1 = 1- real(sqrt(normX^2 + norm(PR)^2 - 2 * innerprod(P,PR)))/normX
                    end
                else
                    
                    if (sum(Jrr) > Rfold) && (nnew ~= folddim(end))
                        %                 Rfold2 = min(Rfold+10,sum(Jrr));
                        %                 fprintf('Rank-%d to Rank-%d CPD\n',sum(Jrr),Rfold);
                        %
                        %                 % Estimate rank-R CPD from rank-(J1+J2+..+JR) CP tensor
                        %                 % Expand factors
                        %                 actdim = find(~cellfun(@isempty,U));
                        %                 param_struct.M = M;
                        %                 param_struct.Jr = Jrr;
                        %                 [diffdim,dimid] = setdiff(actdim,idnnew);
                        %                 param_struct.inmodes = dimid;
                        %
                        %
                        %                 UJ = U(actdim);
                        %                 cp_paramstruct = cp_param;
                        %                 cp_paramstruct.tol = min(1e-6,cp_param.tol);
                        %                 cp_paramstruct.TraceMSAE = false;
                        %                 cp_paramstruct.init = 'random';
                        %                 PR = cpstruct_als(UJ,param_struct,Rfold2,cp_paramstruct); % need linesearch of fLM
                        %                 PR = arrange(PR);
                        %                 U(actdim) = PR.U;
                        %                 U{actdim(1)} = U{actdim(1)} * diag(PR.lambda);
                        
                        %
                        fprintf('Rank-%d to Rank-%d CPD\n',sum(Jrr),Rfold);
                        recontructtype = 'lowrank';
                        
                        % Estimate rank-R CPD from rank-(J1+J2+..+JR) CP tensor
                        % Expand factors
                        actdim = find(~cellfun(@isempty,U));
                        param_struct.M = M;
                        param_struct.Jr = Jrr;
                        [diffdim,dimid] = setdiff(actdim,idnnew);
                        param_struct.inmodes = dimid;
                        
                        
                        UJ = U(actdim);
                        cJr = [0;cumsum(Jrr(:))];
                        UR = cell(numel(actdim),1);
                        for n1 = 1:numel(diffdim)
                            UR{dimid(n1)} = U{diffdim(n1)}(:,1:Rfold);
                        end
                        for n1 = idnnew
                            UR{n1} = U{n1}(:,cJr(1:Rfold)+1);
                        end
                        
                        cp_paramstruct = cp_param;
                        cp_paramstruct.init = {UR 'nvec' 'random' 'random'};
                        cp_paramstruct.tol = min(1e-8,cp_param.tol);
                        cp_paramstruct.TraceMSAE = false;
                        cp_paramstruct.printitn = 0;
                        PR = param.cpstruct_func(UJ,param_struct,Rfold,cp_paramstruct);
                        
                        
                        % Check approximate errors of rank-1 and low-rank tensors
                        Ixnew = cellfun(@(x) size(x,1),UR); Xnbk = reshape(Xnbk,Ixnew(:)');
                        Pr1 = ktensor(UR);
                        %errLR = norm(PR)^2 - 2* innerprod(Xn,PR);
                        %errR1 = norm(Pr1)^2 - 2* innerprod(Xn,Pr1);
                        %if errLR > errR1
                        %    PR = Pr1;
                        %end
                        derr = real(norm(PR)^2 - norm(Pr1)^2 - 2 * real(innerprod(Xnbk,PR-Pr1)));
                        if derr > 0
                            PR = Pr1;
                        end
                        % PJ = ktensor(UJ,Weights); PJ = arrange(PJ);
                        % PR = ktensor(PJ.lambda(1:R),cellfun(@(x) x(:,1:R),PJ.u,'uni',0));
                        % cp_param.init = PR;
                        %PR = cp_eval(PJ,R,param.cp_reffunc,cp_param);
                        PR = arrange(PR);
                        U(actdim) = PR.U;
                        U{actdim(1)} = U{actdim(1)} * diag(PR.lambda);
                        
                    elseif (sum(Jrr) > R)
                        fprintf('Rank-%d to Rank-%d CPD\n',sum(Jrr),R);
                        recontructtype = 'lowrank';
                        
                        % Estimate rank-R CPD from rank-(J1+J2+..+JR) CP tensor
                        % Expand factors
                        actdim = find(~cellfun(@isempty,U));
                        param_struct.M = M;
                        param_struct.Jr = Jrr;
                        [diffdim,dimid] = setdiff(actdim,idnnew);
                        param_struct.inmodes = dimid;
                        
                        
                        UJ = U(actdim);UR = cell(numel(actdim),1);
                        cJr = [0;cumsum(Jrr(:))];
                        for n1 = 1:numel(diffdim)
                            UR{dimid(n1)} = U{diffdim(n1)}(:,1:R);
                        end
                        for n1 = idnnew
                            UR{n1} = U{n1}(:,cJr(1:R)+1);
                        end
                        
                        cp_paramstruct = cp_param;
                        cp_paramstruct.init = {UR 'nvec' 'random'};
                        cp_paramstruct.tol = min(1e-8,cp_param.tol);
                        cp_paramstruct.TraceMSAE = false;
                        cp_paramstruct.printitn = 0;
                        PR = param.cpstruct_func(UJ,param_struct,R,cp_paramstruct);
                        
                        % Check approximate errors of rank-1 and low-rank tensors
                        Ixnew = cellfun(@(x) size(x,1),UR); Xnbk = reshape(Xnbk,Ixnew(:)');
                        Pr1 = ktensor(UR);
                        derr = norm(PR)^2 - norm(Pr1)^2 - 2 * real(innerprod(Xnbk,PR-Pr1));
                        if derr > 0
                            PR = Pr1;
                        end
                        
                        
                        % Check approximate errors of rank-1 and low-rank tensors
                        %                         Ixnew = cellfun(@(x) size(x,1),UJ); Xnbk = reshape(Xnbk,Ixnew(:)');
                        %                         Pr1 = ktensor(U);
                        %                         err = norm(Xnbk)^2 + norm(Pr1)^2 - 2 * real(innerprod(Xnbk,Pr1));
                        
                        
                        %                         M = [];
                        %                         for r = 1:10
                        %                             M = blkdiag(M, ones(1,param_struct.Jr(r)));
                        %                         end
                        %                         % Ktensor Fact
                        %                         for k = 1:numel(param_struct.inmodes)
                        %                             UJ{param_struct.inmodes(k)} = UJ{param_struct.inmodes(k)} * M;
                        %                         end
                        %                         PJ = ktensor(UJ);
                        %                         norm(Xnbk)^2 + norm(PJ)^2 - 2 * real(innerprod(Xnbk,PJ))
                        %
                        %                         PR = cp_fastals(full(PJ),R,struct('init',{{Pr1 'nvecs' 'random'}},'printitn',1));
                        
                        
                        
                        %                 PJ = ktensor(UJ,Weights); PJ = arrange(PJ);
                        %                 PR = ktensor(PJ.lambda(1:R),cellfun(@(x) x(:,1:R),PJ.u,'uni',0));
                        %                 cp_param.init = PR;
                        %PR = cp_eval(PJ,R,param.cp_reffunc,cp_param);
                        PR = arrange(PR);
                        U(actdim) = PR.U;
                        U{actdim(1)} = U{actdim(1)} * diag(PR.lambda);
                    end
                end
                
                t4 = toc(tsrec);
                %nonfolddim = [nonfolddim orderidx{nnew}];
                Nnew = Nnew + numel(orderidx{nnew})-1;
                
                
                %         loc = cellfun(@isempty,U);PR = ktensor(U(~loc));
                %         P = reshape(double(full(X)),size(PR));P = tensor(P);
                %         fitlr = 1- real(sqrt(normX^2 + norm(PR)^2 - 2 * innerprod(P,PR)))/normX;
                %         disp(fitlr)
                
                %         if sum(Jrr) > R
                %             blkname = 'lowrank';
                %         else
                %            blkname = 'Rank1';
                %         end
                %         P = ktensor(U(:),Weights);
                %         fitr1 = 1- real(sqrt(normX^2 + norm(P)^2 - 2 * innerprod(X,P)))/normX;
                %
                %         BagofOut{nnew} = struct('Name',blkname,'Time',t4,'Fit',fitr1,'Level',1);
                
            else % for folding multiple modes (more than 2 modes at a time)
                
                if strcmp(param.foldingreconstruction_type,'sequential')
                    % for folding more than 2 modes - Sequentially reconstruct folding factors
                    if nnew < folddim(end)
                        R2 = Rfold;
                    else
                        R2 = R;
                    end
                    idnnewcur = idnnew;
                    idnnew2 = idnnewcur(1:2);Nnew2 = Nnew;
                    orderidx2 = orderidx;
                    orderidx2{nnew} = idnnew2;
                    
                    orderidx3 = {};
                    for kn = 1:nnew-1
                        orderidx3 = [orderidx3 mat2cell(orderidx2{kn},1,ones(1,numel(orderidx2{kn})))];
                    end
                    nnew2 = numel(orderidx3)+1;
                    orderidx3 = [orderidx3 orderidx2(nnew:end)];
                    
                    % correct order of factors
                    U2 = U;
                    U2(idnnew(3:end)) = [];
                    for kn = nnew2+1:numel(orderidx2)
                        orderidx3{kn} = orderidx3{kn} - numel(idnnewcur)+2;
                    end
                    
                    Ifold = [];
                    for kf = 1:numel(U)
                        if ~isempty(U{kf})
                            Ifold = [Ifold size(U{kf},1)];
                        end
                    end
                    
                    In2 = [Ifold(1:nnew2-1) ...
                        In(idnnew(1)) prod(In(idnnewcur(2:end)))...
                        Ifold(nnew2+1:end)];
                    
                    nonfolddim2 = setdiff(1:Nnew2,nnew2);
                    
                    
                    %         for krec = 1:numel(idnnew)-1 % each run reconstruction one factor
                    %             U2 = cp_2mode_reconstruction(U2,orderidx2,Nnew2,...
                    %                 nonfolddim2,param,In2,R2,Rfold,cp_param,X,normX);
                    %
                    %             U(1:idnnew(krec+1)) = U2(1:idnnew(krec+1));
                    %             locf = cellfun(@isempty,U(idnnew(krec+1)+1:end));
                    %             locf2 = cellfun(@isempty,U2(idnnew(krec+1)+1:end));
                    %             U(find(~locf)+idnnew(krec+1)) = U2(find(~locf2)+idnnew(krec+1));
                    %             %U{idnnew(krec)} = U2{idnnew(krec)};
                    %             %U{idnnew(krec+1)} = U2{idnnew(krec+1)};
                    %
                    %             if krec < numel(idnnew)-1
                    %                 orderidx3{nnew2+krec-1} = orderidx3{nnew2+krec-1}(1);
                    %                 orderidx3(nnew2+krec+1:end+1) = orderidx3(nnew2+krec:end);
                    %                 orderidx3{nnew2+krec} = idnnew(krec+1:min(numel(idnnew),krec+2));
                    %
                    %                 orderidx2 = orderidx3;
                    %                 %nnew2 = nnew2 + 1;
                    %
                    %
                    %                 In2 = [In2(1:nnew2+krec-1) ...
                    %                     In(idnnew(krec+1)) prod(In(idnnew(krec+2:end)))...
                    %                     In2(nnew2+krec+1:end)];
                    %                 %In2(idnnew(krec+1)) = In(idnnew(krec+1));
                    %                 %In2(idnnew(krec+2)) = prod(In(idnnew(krec+2:end)));
                    %                 Nnew2 = Nnew2 +1;
                    %                 %nonfolddim2 = [nonfolddim2 idnnew(krec)];
                    %                 nonfolddim2 = setdiff(1:Nnew2,nnew2+krec);
                    %
                    %                 U2 = U;
                    %                 U2(idnnew(krec+3:end)) = [];
                    %                 for kn = nnew2+krec+1:numel(orderidx2)
                    %                     orderidx2{kn} = orderidx2{kn} - numel(idnnew)+2;
                    %                 end
                    %             end
                    
                    for krec = 1:numel(idnnewcur)-1 % each run reconstruction one factor
                        [U2,Jrr] = cp_2mode_reconstruction(U2,orderidx3,Nnew2,...
                            nonfolddim2,param,In2,R2,Rfold,cp_param);
                        
                        if krec < numel(idnnewcur)-1
                            orderidx3{nnew2+krec-1} = orderidx3{nnew2+krec-1}(1);
                            orderidx3(nnew2+krec+1:end+1) = orderidx3(nnew2+krec:end);
                            orderidx3{nnew2+krec} = idnnewcur(krec+1:min(numel(idnnewcur),krec+2));
                            
                            In2 = [In2(1:nnew2+krec-1) ...
                                In(idnnewcur(krec+1)) prod(In(idnnewcur(krec+2:end)))...
                                In2(nnew2+krec+1:end)];
                            
                            Nnew2 = Nnew2 +1; nonfolddim2 = setdiff(1:Nnew2,nnew2+krec);
                            
                            % insert empty cells for next reconstruction
                            U2(idnnewcur(krec+2)+1:end+1) = U2(idnnewcur(krec+2):end);
                            U2{idnnewcur(krec+2)} = [];
                            
                        else
                            U(1:idnnewcur(krec+1)) = U2(1:idnnewcur(krec+1));
                            locf = cellfun(@isempty,U(idnnewcur(krec+1)+1:end));
                            locf2 = cellfun(@isempty,U2(idnnewcur(krec+1)+1:end));
                            U(find(~locf)+idnnewcur(krec+1)) = U2(find(~locf2)+idnnewcur(krec+1));
                        end
                    end
                    
                    Nnew = Nnew + numel(idnnewcur)-1;
                    
                    
                elseif strcmp(param.foldingreconstruction_type,'direct')
                    
                    tsrec = tic;
                    Jrr = zeros(Rfold,1);M = [];
                    Unnew = U{idnnew(1)};
                    U{idnnew(1)} = []; % reconstructed later.
                    Rfold = size(Unnew,2);
                    %rho = ones(Rfold,1);
                    for r = 1:Rfold
                        V = Unnew(:,r);
                        V = reshape(V,Innew);
                        normV = norm(V(:))^2 ;
                        
                        %                 % Tucker reconstruction
                        %                 V = tensor(V);
                        %                 r1param = struct('maxiters',1000,'tol',1e-6,'printitn',0);
                        %                 Tr = mtucker_als(V,ones(1,ndims(V)),r1param);
                        %                 %Tr = cp_fLMa_v2(V,1,r1param);
                        %                 varexpl = 1-sqrt(1-Tr.lambda^2/normV); % fit instead of varexpl
                        %                 if varexpl < param.var_thresh
                        %                     Jv = zeros(1,ndims(V));
                        %                     for nv = 1:ndims(V)
                        %                         Vn = double(tenmat(V,nv));
                        %                         Vn = Vn * Vn';
                        %                         sv = eig(Vn);sv = sort(sv,'descend');
                        %                         %varexpl = cumsum(sv);varexpl = varexpl/normV;
                        %                         varexpl = 1-real(sqrt(1-cumsum(sv)/normV)); % fit instead of varexpl
                        %                         Jv(nv) = find(varexpl >= param.var_thresh,1,'first');
                        %                     end
                        %                     Tr = mtucker_als(V,Jv,r1param);
                        %                     %Tr = cp_eval(V,max(Jv),param.cp_func,r1param);
                        %                     if isa(Tr,'ttensor')
                        %                         sv = double(Tr.core(:));
                        %                     elseif isa(Tr,'ktensor')
                        %                         sv = T.lambda;
                        %                     end
                        %                     [~,svix] = sort(abs(sv),'descend');sv = sv(svix);
                        %                     %varexpl = cumsum(sv.^2);varexpl = varexpl/normV;
                        %                     varexpl = 1-sqrt(1-cumsum(sv.^2)/normV); % fit instead of varexpl
                        %                     Jr = find(varexpl >= param.var_thresh,1,'first');
                        %                     if isempty(Jr)
                        %                         Jr = numel(varexpl);
                        %                     end
                        %                     svix = svix(1:Jr);
                        %                     svix = ind2sub_full(Jv,svix);
                        %                     Uv = cell(size(svix,2),1);
                        %                     for kix = 1:size(svix,2)
                        %                         Uv{kix} = Tr.U{kix}(:,svix(:,kix));
                        %                     end
                        %                     Uv{end} = Uv{end} * diag(sv(1:Jr));
                        %                 else
                        %                     Uv = Tr.u; Uv{end} = Uv{end} * diag(Tr.lambda);
                        %                     Jr = 1;
                        %                 end
                        %                 for kn = 1:numel(idnnew)
                        %                     U{idnnew(kn)}(:,end+1:end+Jr) = Uv{kn};
                        %                 end
                        
                        % CP reconstruction
                        V = tensor(V);
                        r1param = struct('maxiters',1000,'tol',1e-6,'printitn',0);
                        Tr = mtucker_als(V,ones(1,ndims(V)),r1param);
                        %Tr = cp_fLMa_v2(V,1,r1param);
                        varexpl = 1-sqrt(1-Tr.lambda^2/normV); % fit instead of varexpl
                        if varexpl < param.var_thresh
                            Jv = zeros(1,ndims(V));
                            for nv = 1:ndims(V)
                                Vn = double(tenmat(V,nv));
                                Vn = Vn * Vn';
                                sv = eig(Vn);sv = sort(sv,'descend');
                                %varexpl = cumsum(sv);varexpl = varexpl/normV;
                                varexpl = 1-real(sqrt(1-cumsum(sv)/normV)); % fit instead of varexpl
                                Jv(nv) = find(varexpl >= param.var_thresh,1,'first');
                            end
                            Tr = mtucker_als(V,Jv,r1param);
                            %Tr = cp_eval(V,max(Jv),param.cp_func,r1param);
                            if isa(Tr,'ttensor')
                                sv = double(Tr.core(:));
                            elseif isa(Tr,'ktensor')
                                sv = Tr.lambda;
                            end
                            [~,svix] = sort(abs(sv),'descend');sv = sv(svix);
                            %varexpl = cumsum(sv.^2);varexpl = varexpl/normV;
                            varexpl = 1-sqrt(1-cumsum(sv.^2)/normV); % fit instead of varexpl
                            Jr = find(varexpl >= param.var_thresh,1,'first');
                            if isempty(Jr)
                                Jr = numel(varexpl);
                            end
                            svix = svix(1:Jr);
                            svix = ind2sub_full(Jv,svix);
                            Uv = cell(size(svix,2),1);
                            for kix = 1:size(svix,2)
                                Uv{kix} = Tr.U{kix}(:,svix(:,kix));
                            end
                            Uv{end} = Uv{end} * diag(sv(1:Jr));
                        else
                            Uv = Tr.u; Uv{end} = Uv{end} * diag(Tr.lambda);
                            Jr = 1;
                        end
                        for kn = 1:numel(idnnew)
                            U{idnnew(kn)}(:,end+1:end+Jr) = Uv{kn};
                        end
                        Jrr(r) = Jr;
                    end
                    
                    if param.recurlevel > 0;
                        if sum(Jrr) > Rfold
                            recontructtype = 'lowrank';
                            fprintf('Rank-%d to Rank-%d CPD\n',sum(Jrr),Rfold);
                            
                            actdim = find(~cellfun(@isempty,U));
                            
                            param_struct.Jr = Jrr;
                            [diffdim,dimid] = setdiff(actdim,idnnew);
                            param_struct.inmodes = dimid;
                            
                            
                            % Estimate rank-R CPD from rank-(J1+J2+..+JR) CP tensor
                            % Expand factors
                            %actdim = find(~cellfun(@isempty,U));
                            %                 for n1 = setdiff(actdim,idnnew)
                            %                     U{n1} = U{n1} * M;
                            %                 end
                            UJ = U(actdim);UR = cell(numel(actdim),1);
                            cJr = [0;cumsum(Jrr(:))];
                            for n1 = 1:numel(diffdim)
                                UR{dimid(n1)} = U{diffdim(n1)}(:,1:Rfold);
                            end
                            for n1 = idnnew
                                UR{n1} = U{n1}(:,cJr(1:Rfold)+1);
                            end
                            
                            cp_param.init = UR;
                            cp_param.tol = min(1e-8,cp_param.tol);
                            %PR = cp_eval(PJ,Rfold,param.cp_reffunc,cp_param);
                            cp_paramstruct.printitn = 0;
                            PR = param.cpstruct_func(UJ,param_struct,Rfold,cp_param);
                            
                            % Check approximate errors of rank-1 and low-rank tensors
                            Ixnew = cellfun(@(x) size(x,1),UR); Xnbk = reshape(Xnbk,Ixnew(:)');
                            Pr1 = ktensor(UR);
                            derr = norm(PR)^2 - norm(Pr1)^2 - 2 * innerprod(Xnbk,PR-Pr1);
                            if derr > 0
                                PR = Pr1;
                            end
                            
                            PR = arrange(PR);
                            U(actdim) = PR.U;
                            U{actdim(1)} = U{actdim(1)} * diag(PR.lambda);
                            
                            %             P = reshape(double(full(X)),size(PR));P = tensor(P);
                            %             fitr1 = 1- real(sqrt(normX^2 + norm(PJ)^2 - 2 * innerprod(P,PJ)))/normX
                            %             fitr1 = 1- real(sqrt(normX^2 + norm(PR)^2 - 2 * innerprod(P,PR)))/normX
                        end
                    else
                        
                        if (sum(Jrr) > Rfold) && (nnew ~= folddim(end))
                            recontructtype = 'lowrank';
                            %                 Rfold2 = min(Rfold+10,sum(Jrr));
                            %                 fprintf('Rank-%d to Rank-%d CPD\n',sum(Jrr),Rfold);
                            %
                            %                 % Estimate rank-R CPD from rank-(J1+J2+..+JR) CP tensor
                            %                 % Expand factors
                            %                 actdim = find(~cellfun(@isempty,U));
                            %                 param_struct.M = M;
                            %                 param_struct.Jr = Jrr;
                            %                 [diffdim,dimid] = setdiff(actdim,idnnew);
                            %                 param_struct.inmodes = dimid;
                            %
                            %
                            %                 UJ = U(actdim);
                            %                 cp_paramstruct = cp_param;
                            %                 cp_paramstruct.tol = min(1e-6,cp_param.tol);
                            %                 cp_paramstruct.TraceMSAE = false;
                            %                 cp_paramstruct.init = 'random';
                            %                 PR = cpstruct_als(UJ,param_struct,Rfold2,cp_paramstruct); % need linesearch of fLM
                            %                 PR = arrange(PR);
                            %                 U(actdim) = PR.U;
                            %                 U{actdim(1)} = U{actdim(1)} * diag(PR.lambda);
                            
                            %
                            fprintf('Rank-%d to Rank-%d CPD\n',sum(Jrr),Rfold);
                            
                            % Estimate rank-R CPD from rank-(J1+J2+..+JR) CP tensor
                            % Expand factors
                            actdim = find(~cellfun(@isempty,U));
                            param_struct.M = M;
                            param_struct.Jr = Jrr;
                            [diffdim,dimid] = setdiff(actdim,idnnew);
                            param_struct.inmodes = dimid;
                            
                            
                            UJ = U(actdim);
                            cJr = [0;cumsum(Jrr(:))];
                            UR = cell(numel(actdim),1);
                            for n1 = 1:numel(diffdim)
                                UR{dimid(n1)} = U{diffdim(n1)}(:,1:Rfold);
                            end
                            for n1 = idnnew
                                UR{n1} = U{n1}(:,cJr(1:Rfold)+1);
                            end
                            
                            cp_paramstruct = cp_param;
                            cp_paramstruct.init = UR;
                            cp_paramstruct.tol = min(1e-8,cp_param.tol);
                            cp_paramstruct.TraceMSAE = false;
                            cp_paramstruct.printitn = 0;
                            PR = param.cpstruct_func(UJ,param_struct,Rfold,cp_paramstruct);
                            
                            % Check approximate errors of rank-1 and low-rank tensors
                            Ixnew = cellfun(@(x) size(x,1),UR); Xnbk = reshape(Xnbk,Ixnew(:)');
                            Pr1 = ktensor(UR);
                            derr = norm(PR)^2 - norm(Pr1)^2 - 2 * innerprod(Xnbk,PR-Pr1);
                            if derr > 0
                                PR = Pr1;
                            end
                            
                            %                 PJ = ktensor(UJ,Weights); PJ = arrange(PJ);
                            %                 PR = ktensor(PJ.lambda(1:R),cellfun(@(x) x(:,1:R),PJ.u,'uni',0));
                            %                 cp_param.init = PR;
                            %PR = cp_eval(PJ,R,param.cp_reffunc,cp_param);
                            PR = arrange(PR);
                            U(actdim) = PR.U;
                            U{actdim(1)} = U{actdim(1)} * diag(PR.lambda);
                            
                        elseif (sum(Jrr) > R)
                            fprintf('Rank-%d to Rank-%d CPD\n',sum(Jrr),R);
                            recontructtype = 'lowrank';
                            % Estimate rank-R CPD from rank-(J1+J2+..+JR) CP tensor
                            % Expand factors
                            actdim = find(~cellfun(@isempty,U));
                            param_struct.M = M;
                            param_struct.Jr = Jrr;
                            [diffdim,dimid] = setdiff(actdim,idnnew);
                            param_struct.inmodes = dimid;
                            
                            
                            UJ = U(actdim);UR = cell(numel(actdim),1);
                            cJr = [0;cumsum(Jrr(:))];
                            for n1 = 1:numel(diffdim)
                                UR{dimid(n1)} = U{diffdim(n1)}(:,1:R);
                            end
                            for n1 = idnnew
                                UR{n1} = U{n1}(:,cJr(1:R)+1);
                            end
                            
                            cp_paramstruct = cp_param;
                            cp_paramstruct.init = UR;
                            cp_paramstruct.tol = min(1e-8,cp_param.tol);
                            cp_paramstruct.TraceMSAE = false;
                            cp_paramstruct.printitn = 0;
                            PR = param.cpstruct_func(UJ,param_struct,R,cp_paramstruct);
                            
                            % Check approximate errors of rank-1 and low-rank tensors
                            Ixnew = cellfun(@(x) size(x,1),UR); Xnbk = reshape(Xnbk,Ixnew(:)');
                            Pr1 = ktensor(UR);
                            derr = norm(PR)^2 - norm(Pr1)^2 - 2 * innerprod(Xnbk,PR-Pr1);
                            if derr > 0
                                PR = Pr1;
                            end
                            
                            %                 PJ = ktensor(UJ,Weights); PJ = arrange(PJ);
                            %                 PR = ktensor(PJ.lambda(1:R),cellfun(@(x) x(:,1:R),PJ.u,'uni',0));
                            %                 cp_param.init = PR;
                            %PR = cp_eval(PJ,R,param.cp_reffunc,cp_param);
                            PR = arrange(PR);
                            U(actdim) = PR.U;
                            U{actdim(1)} = U{actdim(1)} * diag(PR.lambda);
                        end
                    end
                    t4 = toc(tsrec);
                    Nnew = Nnew + numel(idnnew)-1;
                    
                end
            end
        end
        folddim = folddimold;
    end
end

% %%
% function exectime = updatetime(exectime,exectime2)
% % exectime.compress = exectime.compress + exectime2.compress;
% % exectime.cp = exectime.cp + exectime2.cp;
% % exectime.r1aprox = exectime.r1aprox + exectime2.r1aprox;
% % exectime.other = exectime.other + exectime2.other;
%
% % if isfield(exectime,'layer')
% %     exectime.layer(end+1) = exectime2;
% % else
% %     exectime.layer(1) = exectime2;
% % end
%
% if nargin ==2
%     if isfield(exectime2,'layer')
%         exectime.layer = exectime2.layer;
%         %         exectime2 = rmfield(exectime2,'layer');
%         %         exectime.layer{end+1} = exectime2;
%     else
%         exectime.layer = {exectime2};
%     end
%
% elseif nargin == 1
%     if isfield(exectime,'layer')
%         foe = exectime;
%         foe = rmfield(foe,'layer');
%         exectime.layer{end+1} = foe;
%         exectime.compress = 0;
%         exectime.cp = 0;
%         exectime.r1aprox = 0;
%         exectime.ref = 0;
%         exectime.other = 0;
%         for kl = 1:numel(exectime.layer)
%             exectime.compress = exectime.compress + exectime.layer{kl}.compress;
%             exectime.cp = exectime.cp + exectime.layer{kl}.cp;
%             exectime.r1aprox = exectime.r1aprox + exectime.layer{kl}.r1aprox;
%             exectime.other = exectime.other + exectime.layer{kl}.other;
%             exectime.ref = exectime.ref + exectime.layer{kl}.ref ;
%         end
%     end
% end
%
% end

% %%
% function output = updateoutput(output,output2)
% if isfield(output2,'layer')
%     output.layer = output2.layer;
%     output2 = rmfield(output2 ,'layer');
%     output.layer{end+1} = output2;
% else
%     output.layer = {output2};
% end
% end
%% Parse input xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
function param = parseInput(opts)
%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;

param.addParamValue('foldingrule','half',@(x) (iscell(x) || isnumeric(x) || ismember(x,{'one' 'half' 'direct'})));
param.addOptional('refine',1);
param.addParamValue('init','backward',@(x) ismember(x,{'forward' 'backward'}));
param.addParamValue('fullrefine',false,@islogical);
param.addParamValue('mindim',3);
param.addParamValue('foldingreconstruction_type','sequential',@(x) (ismember(x,{'direct' 'sequential'})));

% CP function
param.addOptional('cp_func',@cp_fLMa,@(x) isa(x,'function_handle'));
param.addOptional('cp_reffunc',@cp_fLMa,@(x) isa(x,'function_handle'));
cp_param = cp_fastals;
param.addOptional('cp_param',cp_param);

% Compression
compress_param = inputParser;
compress_param.addParamValue('compress',true)
compress_param.addParamValue('compressrawdata',false)
compress_param.addOptional('maxiters',[])
compress_param.addOptional('tol',[])
compress_param.addOptional('R',[],@(x) (isnumeric(x) || isempt(x)));
compopts = [];
if isfield(opts,'compress_param')
    compopts = opts.compress_param;
end
compress_param.parse(compopts);
compress_param = compress_param.Results;
param.addOptional('compress_param',compress_param);

% low-rank approximation
param.addParamValue('var_thresh',.99);

param.addOptional('Rfold',[]); % final rank of CP

param.addOptional('TraceFit',false);

param.addOptional('recurlevel',0);


% Algorithm for Structured CPD
param.addOptional('cpstruct_func',@cpstruct_als,@(x) ((isequal(x,@cpstruct_lroat)) ||isa(x,'function_handle')));

%param.parse(opts);
param.parse(opts);
param = param.Results;
end

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

% %%
% function U = rnk1_approx(T,I)
% % I = size(T);
% N = numel(I);
% U = cell(N,1);
% eigsopts.disp = 0;
%
% for n = 1:numel(I)-1
%     T = reshape(T,I(n),[]);
%     if I(n) <= prod(I(n+1:end))
%         Y = T*T';
%         [u,d] = eigs(Y,1,'LM', eigsopts);
%     else
%         Y = T'*T;
%         [u,d] = eigs(Y,1,'LM', eigsopts);
%         u = T*u;
%         u = bsxfun(@rdivide,u,sqrt(sum(u.^2)));
%     end
%     T = u'*T;
%     U{n} = u;
% end
% U{N} = T';
% end