function [H,U] = ts_deconv_init(X,R,J,opts)
% Initialization for the rank-1 tensor deconvolution
%
%   R: number of rank-1 terms
%   J: size of patten tensors
% opts.init = 'cpd1', 'cpd2','random' or 'tedia'
%
% REF:
%
% [1] A. -H. Phan , P. Tichavsk?, and A. Cichocki, Low rank tensor
% deconvolution, in IEEE International Conference on Acoustics, Speech and
% Signal Processing (ICASSP), pp. 2169 - 2173, 2015.
%
% TENSOR BOX, v.2015

%% Set algorithm parameters from input or by using defaults
param = inputParser;
param.KeepUnmatched = true;
param.addOptional('init','cpd1',@(x) (iscell(x) || isa(x,'ktensor')||...
    ((numel(x)>3) && ismember(x(1:4),{'rand' 'tedi' 'cpd1' 'cpd2'}) )));
param.addOptional('exec_func',@ts_deconv_rank1_als,@(x) isa(x,'function_handle'));

if ~exist('opts','var'), opts = struct; end
param.parse(opts);param = param.Results;
if nargin == 0
    H = param; return
end

N = ndims(X);

%% Initialization TEDIA to the compressed data using Tucker decomposition
% Tucker Compression to size of RJ x RJ
% init = 'tedia';
% init = 'cpd';
% Stage1: Generate initial point using TEDIA or CPD
init = param.init;
SzX = size(X);

if iscell(init)

    if (numel(init)== 2) && iscell(init{1}) && iscell(init{2}) && ...
            (numel(init{1}) == numel(init{2})) && ...
            all(diff(cell2mat(cellfun(@size,init{1},'uni',0)))==0) && ...
            all(diff(cell2mat(cellfun(@size,init{2},'uni',0)))==0)

        U = init{2}; H = init{1};

    else   % small iteratons to find the best initialization

        %normX = norm(X(:));
        bestfit = 0;Pbest = {};
        for ki = 1:numel(init)
            initk = param.init{ki};
            if iscell(initk) || ...
                    (ischar(initk)  && ismember(initk(1:4), ...
                    {'rand' 'cpd1' 'cpd2' 'tedi'}))  % multi-initialization
                if ischar(initk)
                    cprintf('blue','Init. %d - %s\n',ki,initk)
                else
                    cprintf('blue','Init. %d - %s\n',ki,class(initk))
                end
                
                initparam = param;initparam.maxiters = 10;
                initparam.init = initk;
                [H,U] = ts_deconv_init(X,R,J,initparam);
                [H,U,output] = param.exec_func(X,U,H,initparam);
                
                fitinit = 1- output.Error(end);
                if real(fitinit) > bestfit
                    Pbest = {H,U};
                    bestfit = fitinit;kibest = ki;
                end
            end
        end
        cprintf('blue','Choose the best initial value: %d.\n',kibest);
        H = Pbest{1};
        U = Pbest{2};
        
    end

elseif ischar(init)
    
    switch init
        case 'random'
            U = cell(N,1);
            for n = 1:N
                U{n} = randn(SzX(n),J(n),R);
            end 
            
        case 'tedia'
            if J(N) ~= 1
                Xc = mtucker_als(tensor(X),R*max(J)*ones(1,N),struct('dimorder',1:N));
                Utd = cell(N,1);
                
                [Utd{1} Utd{2} Utd{3} St G xblk iter]=tedia4R(double(Xc.core),100);
                Utd = cellfun(@(x,y) x*y, Xc.U,Utd,'uni',0);
                
                
            else
                % Tucker Compression to size of RJ x RJ
                Xc = mtucker_als(tensor(X),R*max(J)*ones(1,N-1),struct('dimorder',1:N-1));
                
                % TEDIA to the compressed tensor Xc.core
                Utd = cell(N-1,1);
                [Utd{1} Utd{2} St xblk iter]=tedia2P(double(Xc.core)/max(abs(double(Xc.core(:)))),1000);
                Utd = cellfun(@(x,y) x*real(y),Xc.U(1:N-1),Utd,'uni',0);
            end
            
            
            % Sts = sum(abs(St),4);
            % Visualize the diagonal block structure
            %         figure(1); clf;
            %         imagesc(xblk)
            
            
            % Generate Toeplitz matrices from outcomes of TEDIA
            U = cell(N,1);
            for n = 1:N
                if J(n) ~= 1
                    U{n} = reshape(Utd{n},size(Utd{n},1),[],R);
                    for r = 1:R
                        [u_est,Sr_est,s_est,err] = toeplitz_factor(U{n}(:,:,r).');
                        U{n}(:,:,r) = Sr_est.';
                    end
                    U{n} = U{n}(:,1:J(n),:);
                end
            end
            
        case 'cpd1'
            
            % CPD -based initialization
            opts = cp_fLMa;
            opts.init = {'dtld' 'nvec' 'random' 'random' 'random'};
            opts.maxiters = 1000;
            opts.printitn = 1;
            if R<=20
                Pcp = cp_fLMa(tensor(squeeze(sum(double(X),N+1))),R,opts);
            else
                opts.linesearch = true;
                Pcp = cp_fastals(tensor(squeeze(sum(double(X),N+1))),R,opts);
            end
            
            SzX = size(X);
            U = cell(N,1);
            for n = 1:N
                U{n} = zeros(SzX(n),J(n),R);
                for r = 1:R
                    U{n}(:,:,r) = convmtx(Pcp.u{n}(1:SzX(n)-J(n)+1,r),J(n));
                end
            end


        case 'cpd2'
            % CPD -based initialization
            opts = cp_fLMa;
            opts.init = {'dtld' 'nvec' 'random' 'random' 'random'};
            opts.maxiters = 1000;
            opts.printitn = 1;
            
            if R*max(J)<30
                Pcp = cp_fLMa(tensor(squeeze(sum(double(X),N+1))),R*max(J),opts);
            else
                opts.linesearch = true;
                Pcp = cp_fastals(tensor(squeeze(sum(double(X),N+1))),R*max(J),opts);
            end

            %Pcp = cp_fLMa(tensor(squeeze(sum(double(X),N+1))),R*max(J),opts);
            [Perm,P] = perm_vectranst(max(J),R);
            Ucp = cellfun(@(x) x(:,Perm),Pcp.u,'uni',0);
            
            % Generate Toeplitz matrices from outcomes of TEDIA
            U = cell(N,1);
            for n = 1:N
                if J(n) ~= 1
                    U{n} = reshape(Ucp{n},size(Ucp{n},1),[],R);
                    for r = 1:R
                        [u_est,Sr_est,s_est,err] = toeplitz_factor(U{n}(:,:,r).');
                        U{n}(:,:,r) = Sr_est.';
                    end
                    U{n} = U{n}(:,1:J(n),:);
                end
            end
            
    end
    
    % Estimate Hr
    if J(N) ~= 1
        LN = prod(J);
        ZZ2 = zeros(LN*R,LN*R);
        ZY = zeros(LN*R,1);
        for r = 1:R
            for s = r:R
                temp = U{1}(:,:,r)'*U{1}(:,:,s);
                for n = 2:N
                    temp = kron(U{n}(:,:,r)'*U{n}(:,:,s),temp);
                end
                ZZ2(LN*(r-1)+1:LN*r,LN*(s-1)+1:LN*s) = temp;
                ZZ2(LN*(s-1)+1:LN*s,LN*(r-1)+1:LN*r) = temp';
            end
            
            Ur = cellfun(@(x) x(:,:,r), U,'uni',0);
            zy = ttm(tensor(X),Ur,1:N,'t');
            zy = reshape(double(zy),LN,[]);
            
            ZY(LN*(r-1)+1:LN*r,:) = zy;
        end
        HH = pinv(ZZ2)*ZY;  % RLP*I% HH = ZZ \ YH;  % RLP*I
        H = tensor(HH', [1 J R]);
        
    else
        
        LN = prod(J);
        ZZ2 = zeros(LN*R,LN*R);
        ZY = zeros(LN*R,SzX(N));
        for r = 1:R
            for s = r:R
                temp = U{1}(:,:,r)'*U{1}(:,:,s);
                for n = 2:N-1
                    temp = kron(U{n}(:,:,r)'*U{n}(:,:,s),temp);
                end
                ZZ2(LN*(r-1)+1:LN*r,LN*(s-1)+1:LN*s) = temp;
                ZZ2(LN*(s-1)+1:LN*s,LN*(r-1)+1:LN*r) = temp';
            end
            
            Ur = cellfun(@(x) x(:,:,r),U(1:N-1),'uni',0);
            zy = ttm(tensor(X),Ur,1:N-1,'t');
            
            ZY(LN*(r-1)+1:LN*r,:) = reshape(double(zy),[],SzX(N));
        end
        HH = pinv(ZZ2)*ZY;
        
        HH = reshape(HH',[SzX(N) prod(J) R]);
        H = zeros(prod(J),R);
        for r = 1:R
            [u,s,v] = svds(HH(:,:,r),1);
            U{N}(:,1,r) = u;
            H(:,r) = s*v;
        end
        
        H = tensor(H, [1 J R]);
    end
end
end