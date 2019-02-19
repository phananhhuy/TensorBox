function [Xp,Err] = b2d_rR(Y,R,opts)
% ALS algorithm to decompose tensor Y into two block terms
%
% Y = G1 x {U1} + G2 x {U2};
% where G1 is of size R1 x R1 x ...x R1
% and G2 is of size R2 x R2 x ... x R2
% Assume R2 > R1
%
%
% % TENSOR BOX, v1. 2014
% Copyright 2014, Phan Anh Huy.

%% Set algorithm parameters from input or by using defaults
if ~exist('opts','var'), opts = struct; end

param = inputParser;
param.KeepUnmatched = true;

param.addOptional('init','block',@(x) (iscell(x) ||...
    ismember(x,{'ajd' 'block' 'td' 'tdWN' 'dtld'})));
param.addOptional('updateterm',false);
param.addOptional('maxiters',1000);
param.addOptional('updatefactor',true);
param.addOptional('updateAG','none',@(x) ismember(x,{'none' 'als' 'svd'}));
param.addOptional('rotatefactor',false);
param.addOptional('correctfactor',false);
param.addOptional('correcttwofactors',false);
param.addOptional('alsupdate',false);
param.addOptional('linesearch','lsb',@(x) ismember(x,{'none' 'lsh' 'lsb' 'elsr' 'elsc'}));

param.addOptional('tol',1e-6);
param.addOptional('printitn',0);

param.parse(opts);
param = param.Results;

if nargin == 0
    Xp = param; return
end


N = ndims(Y);
I = size(Y);
[foe,ix] = sort(sum(R,2));
R = R(ix,:);
P = 2; % == 2 only allow two blocks

%% Initialization
Xp = bcd_init; 

normY = norm(Y);
if isa(Xp{1},'ktensor')
    abserror = (norm(Y-full(Xp{1})-full(Xp{2})));
else
    abserror = Frobnorm(Xp,normY);
end
err = abserror/normY;
errold = err;
Err = err;

% sta = [0 ;prod(R,2)];
eval_fastcost = true;

if ~strcmp(param.linesearch,'none')
    X2 = double(tenmat(Y,3))';
end
%%
done = false; cnt1 = 0; cntdone = 0;maxndone = 5;
while ~done && ( cnt1 < param.maxiters)
    cnt1 = cnt1+1;
    if (cnt1>1) && param.updatefactor && ~strcmp(param.linesearch,'none')
        Xpoldls = Xpold;
    end
    Xpold = Xp;
    
    switch param.updateAG
        case 'als'
            Xp = updateAG(Y,Xp);
        case 'svd'
            Xp = updateAG_svd0(Y,Xp);
            %         Xp = updateAG_svd(Y,Xp);
    end
    
    % Alternating update Factors and core tensors
    if param.updatefactor
%         if (cnt1>1)  && ~strcmp(param.linesearch,'none')
%             % Perform Line Search
%             dUp1 = cellfun(@(x,y) x-y,Xp{1}.U,Xpoldls{1}.U,'uni',0);
%             dUp2 = cellfun(@(x,y) x-y,Xp{2}.U,Xpoldls{2}.U,'uni',0);
%             
%             A2_cell = {Xp{1}.U{1} Xp{2}.U{1}};
%             B2_cell = {Xp{1}.U{2} Xp{2}.U{2}};
%             C2_cell = {Xp{1}.U{3} Xp{2}.U{3}};
%             
%             dA_cell = {dUp1{1} dUp2{1}};  % search direction for A
%             dB_cell = {dUp1{2} dUp2{2}};  % search direction for B
%             dC_cell = {dUp1{3} dUp2{3}};  % search direction for C
%             
%             if R(2,1) >1
%                 D_cell = {Xp{1}.lambda  double(Xp{2}.core)};
%             else
%                 D_cell = {Xp{1}.lambda  double(Xp{2}.lambda)};
%             end
%             
%             it1 = 5;
%             [A_cell,B_cell,C_cell] = bcdLrMrNr_lsearch(A2_cell,B2_cell,C2_cell,...
%                 dA_cell,dB_cell,dC_cell,D_cell,...
%                 X2,param.linesearch,it1,R(:,1)',R(:,1)',R(:,1)');
% 
% %             [A_cell,B_cell,C_cell,D_cell] = bcdLrMrNr_lsearch2(A2_cell,B2_cell,C2_cell,...
% %                 dA_cell,dB_cell,dC_cell,D_cell,...
% %                 X2,param.linesearch,it1,R(:,1)',R(:,1)',R(:,1)',normY);
%             
%             Xp{1}.U{1} = A_cell{1};Xp{2}.U{1} = A_cell{2};
%             Xp{1}.U{2} = B_cell{1};Xp{2}.U{2} = B_cell{2};
%             Xp{1}.U{3} = C_cell{1};Xp{2}.U{3} = C_cell{2};
%             
% %             Xp{1}.lambda = D_cell{1};
% %             if R(2,1) >1
% %                 Xp{2}.core = tensor(D_cell{2});
% %             else
% %                 Xp{2}.lambda = D_cell{2};
% %             end
%             
% %             Xp = fastupdatecore(Y,Xp);
%             Xp = rotatefactor(Xp);
% %             if P == 2
% %                 Xp = fastupdatecore(Y,Xp);
% %             else
%                 Xp = updatecore(Y,Xp);
% %             end
%             
%         end
%         if R(1,1)==1
%             Xp = als_rank1R_v2(Y,Xp); %Xp = als_rank1R(Y,Xp);
%         else
% %             Xp= alsupdate(Y,Xp); % use line search
%             %Xp = fastals(Y,Xp);
%             Xp = fastals_blk(Y,Xp);
%             % Xp = als_procrustes(Y,Xp);
%         end

        
          if strcmp(param.linesearch,'none')
              if R(1,1)==1
                  Xp = als_rank1R_v2(Y,Xp); %Xp = als_rank1R(Y,Xp);
              else
                  % Xp= alsupdate(Y,Xp); % use line search
                  %Xp = fastals(Y,Xp);
                  Xp = fastals_blk(Y,Xp);
                  % Xp = als_procrustes(Y,Xp);
              end
          else
              Xp= alsupdate(Y,Xp); % use line search
          end
    end
    
    % Alternating rules update block terms
    if param.updateterm
        eval_fastcost = false;
        Yh = 0;
        for p = 1:P
            Yh = Yh + tensor(Xp{p});
        end
        
        
        Yr = Y - Yh;
        for p = 1:P
            Yr = Yr + tensor(Xp{p});
            Xp{p} = mtucker_als(Yr,R(p,:),opt_hooi,Xp{p}.U);
            Yr = Yr - tensor(Xp{p});
        end
        
        %% update core
        %Xp = rotatefactor(Xp);
        %Xp = fastupdatecore(Y,Xp);
    end
    
    
    %% rotate An such that A1n^T * A2n = diag(s) based on Procrustes
    % Fix one factor A2 and update other one
    % A1 = A2 * diag(s) + A2c * Q *diag(sqrt(1-s^2)))
    if param.correctfactor
        [Xp,err] = correctfactor(Xp,err);
    end
    
    %% Correct two factors rotate An such that A1n^T * A2n = diag(s)
    if param.correcttwofactors
        [Xp,err] = correcttwofactors(Xp,err);
    end
    
     
    %% Fast Approximate error
    if eval_fastcost
        abserror = Frobnorm(Xp,normY);
    else
        abserror = (norm(Y-full(Xp{1})-full(Xp{2})));
    end
    
    err = abserror/normY;
    
    Err = [Err err];
    
    if mod(cnt1,param.printitn) == 0
        fprintf('Iter %d, Err %d\n',cnt1,err)
    end
    
    if ~isinf(errold)
        done = (abs(err-errold) <= err * param.tol) ; %|| (abs(err-errold) < param.tol);
        
        if done 
            cntdone = cntdone +1;
            if cntdone < maxndone
                done = false;
            end
        else
            done = false;
            cntdone = 0;
        end
        
    end
    errold = err;
    
end

% Rotate factors
Xp = rotatefactor(Xp);

if P == 2
    if N == 3
%         ah = Xp{1}.u; Uh = Xp{2}.u;
%         u = cellfun(@(x) x(:,1),Uh,'uni',0);
%         rho = real(cellfun(@(x,y) x'*y(:,1),ah,Uh));
%         rhoall = prod(rho);
%         ldah = (ttv(Y,ah) - rhoall*ttv(Y,u))/(1-rhoall^2);
%         
%         G = ttm(Y,Uh,'t');G = G.data;
%         G(1) = G(1) - ldah *rhoall;
%         G = tensor(G);
%         
%         Xp{1} = ktensor(ldah,ah);
%         if size(Uh{1},2)> 1
%             Xp{2} = ttensor(G,Uh);
%         else
%             Xp{2} = ktensor(double(G),Uh);
%         end
            
        Xp = fastupdatecore(Y,Xp);
    elseif N == 4
        Xp = fastupdatecore_4way(Y,Xp);
    end
else
    Xp = updatecore(Y,Xp);
end

% end of the main algorithm

%% *********************************************************************
    function abserror = Frobnorm(Xp,normY)
        % Fast Computation of the Frobenius norm of the residual tensor Y -
        % Yhat
        % normY = norm(Y) is the Frobenius norm of Y.
        % Using with least squares algorithm, e.g., ALS
        
        %% Approximate error
        if R(1,1) == 1
            % Fast cost when the first block has rank 1
            q = real(prod(cellfun(@(x,y) x'*y(:,1),Xp{1}.U,Xp{2}.U)));
            if isa(Xp{2},'ttensor')
                abserror = normY^2 - abs(Xp{1}.lambda)^2 - norm(Xp{2}.core)^2 - ...
                    2 * q*real(Xp{1}.lambda'*Xp{2}.core(1));
            else
                abserror = normY^2 - (Xp{1}.lambda)^2 - (Xp{2}.lambda)^2 - ...
                    2 *q* real(Xp{1}.lambda.'*Xp{2}.lambda);
            end
        else
            UtV = cell(N,1);
            for n = 1:N
                UtV{n} = sum(Xp{1}.U{n}.*Xp{2}.U{n}(:,1:R(1,n)))';
            end
%                 UtV = {sum(Xp{1}.U{1}.*Xp{2}.U{1}(:,1:R(1,1)))' ...
%                     sum(Xp{1}.U{2}.*Xp{2}.U{2}(:,1:R(1,2)))' ...
%                     sum(Xp{1}.U{3}.*Xp{2}.U{3}(:,1:R(1,3)))'};
            H = full(ktensor(UtV));
            
            if N == 3
                abserror = normY^2 -norm(Xp{1}.core)^2 - norm(Xp{2}.core)^2 - ...
                    2 * innerprod(Xp{1}.core,Xp{2}.core(1:R(1,1),1:R(1,2),1:R(1,3)).*H);
            elseif N == 4
                abserror = normY^2 -norm(Xp{1}.core)^2 - norm(Xp{2}.core)^2 - ...
                2 * innerprod(Xp{1}.core,Xp{2}.core(1:R(1,1),1:R(1,2),1:R(1,3),1:R(1,4)).*H);
            end
                
            
        end
        abserror = real(sqrt(abserror));
    end

%% ********************************************************************
    function Xp = rotatefactor(Xp)
        % Rotate factors to be orthogonal and decorrelate
        % Rotate factors
        for p2 = 1:P
            for n =1:N
                [temp,rr] = qr(Xp{p2}.U{n},0);
                Xp{p2}.U{n} = temp;
                if isa(Xp{p2},'ttensor')
                    Xp{p2}.core = ttm(Xp{p2}.core,rr,n);
                elseif isa(Xp{p2},'ktensor')
                    Xp{p2}.lambda = Xp{p2}.lambda * rr;
                end
            end
        end
        % Rotate factors to be orthogonal
        for n2 = 1:N
            [u,s,v] = svd(Xp{1}.U{n2}'*Xp{2}.U{n2},0);
            
            Xp{1}.U{n2} = Xp{1}.U{n2} * u;
            Xp{2}.U{n2} = Xp{2}.U{n2} * v;
            
            if isa(Xp{1},'ttensor')
                Xp{1}.core = ttm(Xp{1}.core,u',n2);
            elseif isa(Xp{1},'ktensor')
                Xp{1}.lambda = Xp{1}.lambda * u';
            end
            
            
            if isa(Xp{2},'ttensor')
                Xp{2}.core = ttm(Xp{2}.core,v',n2);
            elseif isa(Xp{2},'ktensor')
                Xp{2}.lambda = Xp{2}.lambda * v';
            end
        end
    end


%% ********************************************************************
    function Xp = updatecore(Y,Xp)
        % Least squares algorithm updates core tensors
        % Suggest using the fast_updatecore
        % Update core tensors
        W =[];
        for p2 = 1:P
            KRN = kron(Xp{p2}.U{N},Xp{p2}.U{N-1});
            for kn = N-2:-1:1
                KRN = kron(KRN,Xp{p2}.U{kn});
            end
            W = [W KRN];
        end
        G = W\reshape(Y.data,I(1)*I(2)*I(3),1);
        
        G = mat2cell(G,prod(R,2),1);
        for p2 = 1:P
            if prod(R(p2,:),2) ==1
                Xp{p2}.lambda = G{p2};
            else
                Xp{p2}.core = tensor(reshape(G{p2},R(p2,:)));
            end
        end
    end


%     function Xp = fastupdatecore_old(Y,Xp)
%         % Call after rotate factors to be independent
%         % update core
%         %G = W\reshape(Y.data,I(1)*I(2),I(3));
%
%         UtV = kron(sum(Xp{1}.U{3}.*Xp{2}.U{3}(:,1:R(1,3))),...
%             kron(sum(Xp{1}.U{2}.*Xp{2}.U{2}(:,1:R(1,2))),...
%             sum(Xp{1}.U{1}.*Xp{2}.U{1}(:,1:R(1,1)))));
%
%         G = [reshape(double(ttm(Y,Xp{1}.U,'t')),prod(R(1,:)),1)
%             reshape(double(ttm(Y,Xp{2}.U,'t')),prod(R(2,:)),1)];
%
%         % fast inverse
%         %WtW = [eye(prod(R(1,:))) diag(UtV);
%         %    diag(UtV) eye(prod(R(2,:)))];
%         %G = WtW\G;
%         iWtW = [diag(1./(1-UtV.^2)) -diag(UtV./(1-UtV.^2));
%            -diag(UtV./(1-UtV.^2))   diag(1./(1-UtV.^2))];
%         iWtW = blkdiag(iWtW,eye(prod(R(2,:))-prod(R(1,:))));
%
%         G = iWtW * G;
%
%         G = mat2cell(G,prod(R,2),1);
%         for p2 = 1:P
%             if isa(Xp{p2},'ttensor')
%                 Xp{p2}.core = tensor(reshape(G{p2},R(p2,:)));
%             else
%                 Xp{p2}.lambda = G{p2};
%             end
%         end
%
%     end

%     function Xp = fastupdatecore(Y,Xp)
%         % Update core tensors of block terms
%         % Call after rotate factors to be independent
%         % update core
%         %G = W\reshape(Y.data,I(1)*I(2),I(3));
%
%         UtV = {sum(Xp{1}.U{1}.*Xp{2}.U{1})' sum(Xp{1}.U{2}.*Xp{2}.U{2})' ...
%             sum(Xp{1}.U{3}.*Xp{2}.U{3})'};
%
%         G1 = ttm(Y,Xp{1}.U,'t');
%         G2 = ttm(Y,Xp{2}.U,'t');
%
%         G3 = full(ktensor(UtV));
%
%         if isa(Xp{1},'ttensor')
%             Xp{1}.core = (G1 - G2.*G3)./(1-G3.^2+eps);
%         elseif isa(Xp{1},'ktensor')
%             Xp{1}.lambda = double((G1 - G2.*G3)./(1-G3.^2+eps));
%         end
%
%         if isa(Xp{2},'ttensor')
%             Xp{2}.core = (G2 - G1.*G3)./(1-G3.^2+eps);
%         elseif isa(Xp{2},'ktensor')
%             Xp{2}.lambda = double((G2 - G1.*G3)./(1-G3.^2+eps));
%         end
%     end

%% ********************************************************************
    function Xp = fastupdatecore2(Y,U,V)
        % Update core tensors of block terms
        % Call after rotate factors to be independent
        % update core
        %G = W\reshape(Y.data,I(1)*I(2),I(3));
        
        UtV = {sum(U{1}.*V{1}(:,1:R(1,1)))' ...
            sum(U{2}.*V{2}(:,1:R(1,2)))' ...
            sum(U{3}.*V{3}(:,1:R(1,3)))'};
        
        G1 = (ttm(Y,U,'t'));
        G2 = (ttm(Y,V,'t'));
        
        G3 = (full(ktensor(UtV)));
        
        if size(U{1},2) > 1
            temp = (G1 - G2(1:R(1,1),1:R(1,2),1:R(1,3)).*G3)./(1-G3.^2+eps);
            Xp{1} = ttensor(temp,U);
        else
            lambda = double((G1 - G2(1:R(1,1),1:R(1,2),1:R(1,3)).*G3)./(1-G3.^2+eps));
            Xp{1} = ktensor(lambda,U);
        end
        
        if size(V{1},2) > 1
            G2(1:R(1,1),1:R(1,2),1:R(1,3)) = (G2(1:R(1,1),1:R(1,2),1:R(1,3)) - G1.*G3)./(1-G3.^2+eps);
            Xp{2} = ttensor(G2,V);
        else
            lambda = double((G2 - G1.*G3)./(1-G3.^2+eps));
            Xp{2} = ktensor(lambda,V);
        end
    end

%% ********************************************************************
    function Xp = fastupdatecore(Y,Xp)
        % Update core tensors of block terms
        % Call after rotate factors to be independent
        % update core
        %G = W\reshape(Y.data,I(1)*I(2),I(3));
        
        
        UtV  = {real(Xp{2}.U{1}(:,1:R(1,1))'*Xp{1}.U{1}) ...
                real(Xp{2}.U{2}(:,1:R(1,2))'*Xp{1}.U{2}) ...
                real(Xp{2}.U{3}(:,1:R(1,3))'*Xp{1}.U{3})}; % the same as rho when R=1
       
        %UtV = {sum(Xp{1}.U{1}.*Xp{2}.U{1}(:,1:R(1,1)))' ...
        %    sum(Xp{1}.U{2}.*Xp{2}.U{2}(:,1:R(1,2)))' ...
        %    sum(Xp{1}.U{3}.*Xp{2}.U{3}(:,1:R(1,3)))'};
        
        G1 = (ttm(Y,Xp{1}.U,'t'));
        G2 = (ttm(Y,Xp{2}.U,'t'));
        
        G3 = (full(ktensor(UtV)));
        G3 = double(G3);
        
        if isa(Xp{1},'ttensor')
            Xp{1}.core = (G1 - G2(1:R(1,1),1:R(1,2),1:R(1,3)).*G3)./(1-abs(G3).^2+eps);
        elseif isa(Xp{1},'ktensor')
            Xp{1}.lambda = double((G1 - G2(1:R(1,1),1:R(1,2),1:R(1,3)).*G3)./(1-abs(G3).^2+eps));
        end
        
        if isa(Xp{2},'ttensor')
            %G2(1) = G2(1) - G3*Xp{1}.lambda;
            G2(1:R(1,1),1:R(1,2),1:R(1,3)) = (G2(1:R(1,1),1:R(1,2),1:R(1,3)) - G1.*G3)./(1-abs(G3).^2+eps);
            Xp{2}.core = G2;
        elseif isa(Xp{2},'ktensor')
            Xp{2}.lambda = double((G2 - G1.*G3)./(1-abs(G3).^2+eps));
        end
        
%         if isa(Xp{2},'ttensor')
%             G2(1:R(1,1),1:R(1,2),1:R(1,3)) = (G2(1:R(1,1),1:R(1,2),1:R(1,3)) - G1.*G3)./(1-G3.^2+eps);
%             Xp{2}.core = G2;
%         elseif isa(Xp{2},'ktensor')
%             Xp{2}.lambda = double((G2 - G1.*G3)./(1-G3.^2+eps));
%         end
    end


% ********************************************************************
    function Xp = fastupdatecore_4way(Y,Xp)
        % Update core tensors of block terms
        % Call after rotate factors to be independent
        % update core
        %G = W\reshape(Y.data,I(1)*I(2),I(3));
        
        UtV = {sum(Xp{1}.U{1}.*Xp{2}.U{1}(:,1:R(1,1)))' ...
            sum(Xp{1}.U{2}.*Xp{2}.U{2}(:,1:R(1,2)))' ...
            sum(Xp{1}.U{3}.*Xp{2}.U{3}(:,1:R(1,3)))' ...
            sum(Xp{1}.U{4}.*Xp{2}.U{4}(:,1:R(1,4)))'};
        
        G1 = (ttm(Y,Xp{1}.U,'t'));
        G2 = (ttm(Y,Xp{2}.U,'t'));
        
        G3 = (full(ktensor(UtV)));
        
        if isa(Xp{1},'ttensor')
            Xp{1}.core = (G1 - G2(1:R(1,1),1:R(1,2),1:R(1,3),1:R(1,4)).*G3)./(1-G3.^2+eps);
        elseif isa(Xp{1},'ktensor')
            Xp{1}.lambda = double((G1 - G2(1:R(1,1),1:R(1,2),1:R(1,3),1:R(1,4)).*G3)./(1-G3.^2+eps));
        end
        
        if isa(Xp{2},'ttensor')
            %G2(1) = G2(1) - G3*Xp{1}.lambda;
            G2(1:R(1,1),1:R(1,2),1:R(1,3),1:R(1,4)) = (G2(1:R(1,1),1:R(1,2),1:R(1,3),1:R(1,4)) - G1.*G3)./(1-G3.^2+eps);
            Xp{2}.core = G2;
        elseif isa(Xp{2},'ktensor')
            Xp{2}.lambda = double((G2 - G1.*G3)./(1-G3.^2+eps));
        end
        
%         if isa(Xp{2},'ttensor')
%             G2(1:R(1,1),1:R(1,2),1:R(1,3)) = (G2(1:R(1,1),1:R(1,2),1:R(1,3)) - G1.*G3)./(1-G3.^2+eps);
%             Xp{2}.core = G2;
%         elseif isa(Xp{2},'ktensor')
%             Xp{2}.lambda = double((G2 - G1.*G3)./(1-G3.^2+eps));
%         end
    end

%% ********************************************************************
    function Xp= alsupdate(Y,Xp)
        % Call Nion' ALS algorithm with line search
        %--- Algorithm parameters
        comp='off';          % ='on' or ='off' to perform or not dimensionality reduction
        Tol1=1e-6;          % Tolerance
        MaxIt1=10;        % Max number of iterations
        Tol2=1e-4;          % tolerance in refinement stage (after decompression)
        MaxIt2=10;         % Max number of iterations in refinement stage
        Ninit=5;
        ls = param.linesearch;%'lsb';
        
        A_init = {};
        B_init = {};
        C_init = {};
        for p = 1:P
            A_init{p} = Xp{p}.U{1};
            B_init{p} = Xp{p}.U{2};
            C_init{p} = Xp{p}.U{3};
        end
        %D_init = {double(Xp{1}.core) double(Xp{2}.core)};
        D_init  = {};
        for p2 = 1:P
            if isa(Xp{p2},'ktensor')
                D_init = [D_init {Xp{p2}.lambda}];
            else
                D_init = [D_init {double(Xp{p2}.core)}];
            end
        end
        [A_est,B_est,C_est,D_est,phi,it1,it2,phi_als]=bcdLrMrNr_alsls(Y.data,...
            R(:,1)',R(:,2)',R(:,3)',ls,comp,Tol1,MaxIt1,Tol2,MaxIt2,Ninit,A_init,B_init,C_init,D_init);
%         [A_est,B_est,C_est,D_est,phi,it1,it2,phi_als]=bcdLrMrNr_alsls_twobcks(Y.data,...
%            R(:,1)',R(:,2)',R(:,3)',ls,comp,Tol1,MaxIt1,Tol2,MaxIt2,Ninit,A_init,B_init,C_init,D_init);
        for p2 = 1:P
            if numel(D_init{p2}) == 1
                Xp{p2} = ktensor(D_est{p2},{A_est{p2}, B_est{p2}, C_est{p2}});
            else
                Xp{p2} = ttensor(tensor(D_est{p2}),{A_est{p2}, B_est{p2}, C_est{p2}});
            end
        end
        
    end


%% *********************************************************************
    function Xp = fastals(Y,Xp)
        % Fast ALS update factors and core tensors
        utu = cellfun(@(x,y) x'*y,Xp{1}.u,Xp{2}.u,'uni',0);
        
        for n2 = 1:N
            
            % Fast update Un
            t1 = ttt(Xp{1}.core,Xp{1}.core,[1:n2-1 n2+1:N]);
            t2 = ttm(Xp{1}.core,utu,-n2,'t');
            t2 = ttt(tensor(t2),Xp{2}.core,[1:n2-1 n2+1:N]);
            t3 = ttt(Xp{2}.core,Xp{2}.core,[1:n2-1 n2+1:N]);
            t4 = [double(t1) double(t2); double(t2)' double(t3)];
            
            t5 = [double(ttt(ttm(Y,Xp{1}.U,-n2,'t'),Xp{1}.core,[1:n2-1 n2+1:N])), ...
                double(ttt(ttm(Y,Xp{2}.U,-n2,'t'),Xp{2}.core,[1:n2-1 n2+1:N]))];
            Un = t5/t4;
            Un = mat2cell(Un,I(n2),R(:,n2)');
            
            % Rotate factors to be orthogonal
            for p2 = 1:P
                Xp{p2}.U{n2} = Un{p2};
                [Xp{p2}.U{n2},rr] = qr(Xp{p2}.U{n2},0);
                
                if isa(Xp{p2},'ttensor')
                    Xp{p2}.core = ttm(Xp{p2}.core,rr,n2);
                elseif isa(Xp{p2},'ktensor')
                    Xp{p2}.lambda= Xp{p2}.lambda*rr;
                end
            end
            
            % decorrelate factors among block terms
            [u,s,v] = svd(Xp{1}.U{n2}'*Xp{2}.U{n2},0);
            Xp{1}.U{n2} = Xp{1}.U{n2} * u;
            Xp{2}.U{n2} = Xp{2}.U{n2} * v;
            
            utu{n2} = s;
            
            if isa(Xp{1},'ttensor')
                Xp{1}.core = ttm(Xp{1}.core,u',n2);
            elseif isa(Xp{1},'ktensor')
                Xp{1}.lambda = Xp{1}.lambda * u';
            end
            
            if isa(Xp{2},'ttensor')
                Xp{2}.core = ttm(Xp{2}.core,v',n2);
            elseif isa(Xp{2},'ktensor')
                Xp{2}.lambda = Xp{2}.lambda * v';
            end
        end
        
        %         %% Check collinearity between terms
        %         utu = cell2mat(cellfun(@(x) diag(x),utu,'uni',0));
        %         congru = abs(prod(utu,2));
        %         ix = find(congru>.95);
        %         U = Xp{1}.U; V = Xp{2}.U;
        %         if ~isempty(ix)
        %             if congru(end-numel(ix)+1)<0.8
        %                 ixc = numel(ix);
        %             else
        %                 ixc = find(congru(end-numel(ix)+1:end)<.8,1);
        %                 ixc = numel(ix)-ixc;
        %             end
        %
        %
        %             for n2 = 1:N
        %                 uh = V{n2}(:,ix); ul = U{n2}(:,end-ixc+1:end);
        %                 U{n2} = [uh U{n2}(:,1:end-ixc)];
        %                 V{n2} = [V{n2}(:,ix+1:end) ul];
        %                 R(1,n2) = size(U{n2},2);
        %                 R(2,n2) = size(V{n2},2);
        %
        %
        %                 % Rotate factors to be orthogonal
        %                 [U{n2},rr] = qr(U{n2},0);
        %                 [V{n2},rr] = qr(V{n2},0);
        %
        %                 % decorrelate factors among block terms
        %                 [u,s,v] = svd(U{n2}'*V{n2},0);
        %                 U{n2} = U{n2} * u;
        %                 V{n2} = V{n2} * v;
        %
        %             end
        %
        %         end
        
        if P == 2
            Xp = fastupdatecore(Y,Xp);
            %             Xp = fastupdatecore2(Y,U,V);
        else
            Xp = updatecore(Y,Xp);
        end
    end

%% *********************************************************************
    function Xp = fastals_blk(Y,Xp)
        % Fast ALS update factors and core tensors
        UtV = cellfun(@(x,y) x'*y,Xp{1}.u,Xp{2}.u,'uni',0);
        
        for n2 = 1:N
            T1 = ttm(Y,Xp{1}.U,-n2,'t');
            if isa(Xp{1},'ttensor')
                T1 = ttt(T1,Xp{1}.core,setdiff(1:N,n2));
            elseif isa(Xp{1},'ktensor')
                T1 = tenmat(T1,n2)* Xp{1}.lambda;
            end
            T1 = double(T1);
            T2 = ttm(Y,Xp{2}.U,-n2,'t');
            
            if isa(Xp{2},'ttensor')
                T2 = ttt(T2,Xp{2}.core,setdiff(1:N,n2));
            elseif isa(Xp{2},'ttensor')
                T2 = tenmat(T2,n2)* Xp{2}.lambda;
            end
            T2 = double(squeeze(T2));
            
            if isa(Xp{1},'ttensor')
                K1 = ttt(Xp{1}.core,Xp{1}.core,setdiff(1:N,n2));
            elseif isa(Xp{1},'ktensor')
                K1 = Xp{1}.lambda*Xp{1}.lambda;
            end
            K1 = double(K1);
            
            if isa(Xp{2},'ttensor')
                K2 = ttt(Xp{2}.core,Xp{2}.core,setdiff(1:N,n2));
            elseif isa(Xp{2},'ktensor')
                K2 = Xp{2}.lambda*Xp{2}.lambda;
            end
            K2 = double(K2);
            
            if isa(Xp{2},'ttensor')
                Cx = ttm(Xp{2}.core,UtV,-n2);
            elseif isa(Xp{2},'ktensor')
                Cx = ktensor(Xp{2}.lambda,UtV);
            end
            Cx = full(Cx);
            if isa(Xp{1},'ttensor')
                Cx = ttt(Cx,Xp{1}.core,setdiff(1:N,n2));
            elseif isa(Xp{1},'ktensor')
                Cx = tenmat(Cx,n2)*Xp{1}.lambda;
            end
            Cx = double(Cx);
            
            
            if R(1,1)/R(2,1)>.8
                iK2Cx = K2\Cx;
                Ls1 = T1 - T2 * iK2Cx;
                Rs1 = K1 - Cx' * iK2Cx;
                
                Un{1} = Ls1/Rs1;
                
                iK1Cxt = K1\Cx';
                Ls2 = T2 - T1 * iK1Cxt;
                Rs2 = K2 - Cx * iK1Cxt;
                Un{2} = Ls2/Rs2;
            else
                iK2 = pinv(K2);
                iK2Cx = iK2*Cx;
                Ls1 = T1 - T2 * iK2Cx;
                Rs1 = K1 - Cx' * iK2Cx;
                Rs1 = pinv(Rs1);
                Un{1} = Ls1 * Rs1;
                
                Ls2 = T2 - T1 * (K1\Cx');
                Rs2 = iK2 +  iK2Cx * Rs1 * iK2Cx';
                
                Un{2} = Ls2 *Rs2;
            end
            
            % Rotate factors to be orthogonal
            for p2 = 1:P
                Xp{p2}.U{n2} = Un{p2};
                [Xp{p2}.U{n2},rr] = qr(Xp{p2}.U{n2},0);
                
                if isa(Xp{p2},'ttensor')
                    Xp{p2}.core = ttm(Xp{p2}.core,rr,n2);
                elseif isa(Xp{p2},'ktensor')
                    Xp{p2}.lambda= Xp{p2}.lambda*rr;
                end
            end
            
            % decorrelate factors among block terms
            [u,s,v] = svd(Xp{1}.U{n2}'*Xp{2}.U{n2},0);
            Xp{1}.U{n2} = Xp{1}.U{n2} * u;
            Xp{2}.U{n2} = Xp{2}.U{n2} * v;
            
            UtV{n2} = s;
            
            if isa(Xp{1},'ttensor')
                Xp{1}.core = ttm(Xp{1}.core,u',n2);
            elseif isa(Xp{1},'ktensor')
                Xp{1}.lambda = Xp{1}.lambda * u';
            end
            
            if isa(Xp{2},'ttensor')
                Xp{2}.core = ttm(Xp{2}.core,v',n2);
            elseif isa(Xp{2},'ktensor')
                Xp{2}.lambda = Xp{2}.lambda * v';
            end
            
            Xp = fastupdatecore(Y,Xp); % the code has not been optimized, can be faster
        end
        
        %        Xp = fastupdatecore(Y,Xp);
    end

%% *********************************************************************
    function [Xp,err] = correctfactor(Xp,err)
        % Fix A2 and Update A1 = A2 diag(s)^2 + A2c *Q *diag(sqrt(1-s^2)
        % Find s and Q where [A2 A2c] are column space of Y2
        
        Xp = rotatefactor(Xp);
        if P == 2
            Xp = fastupdatecore(Y,Xp);
        else
            Xp = updatecore(Y,Xp);
        end
        
        done2 = false;errold2 = errold;cnt = 0;
        while  ~done2 && ( cnt < 20)
            cnt = cnt + 1;
            
            %% Correct A1
            % Fix A2 and Update A1 = A2 diag(s)^2 + A2c *Q *diag(sqrt(1-s^2)
            % Find s and Q where [A2 A2c] are column space of Y2
            
            for n = 1:N
                % Find s
                Zp{1} = ttm(Xp{1}.core,Xp{1}.U,-n);
                Zp{1} = double(tenmat(Zp{1},n));
                
                Zp{2} = ttm(Xp{2}.core,Xp{2}.U,-n);
                Zp{2} = double(tenmat(Zp{2},n));
                
                
                UnYn = Xp{2}.U{n}'*double(tenmat(Y,n)) - Zp{2};
                s = sum(UnYn(1:R(1,n),:).*Zp{1},2)./sum(Zp{1}.*Zp{1},2);
                s = min(max(-1,s),1);
                
                % Find Q
                [qq,rr] = qr(Xp{2}.U{n});
                ix = find(sum(abs(Xp{2}.U{n}'*qq))<1e-8);
                Unbar = qq(:,ix);
                temp = Unbar'*double(tenmat(Y,n))*Zp{1}'*diag(sqrt(1-s.^2));
                [ut,st,vt] = svd(temp);
                Qn = ut*vt';
                
                Xp{1}.U{n} = Xp{2}.U{n}(:,1:R(1,n)) * diag(s) + Unbar * Qn * diag(sqrt(1-s.^2));
            end
            
            % update core
            if P == 2
                Xp = fastupdatecore(Y,Xp);
            else
                Xp = updatecore(Y,Xp);
            end
            
            
            % Correct A2
            % Fix A1 and Update A2 = A1 diag(s)^2 + A1c *Q *diag(sqrt(1-s^2)
            % Find s and Q where [A1 A1c] are column space of Y1
            
            for n = 1:N
                
                % Find s
                Zp{1} = ttm(Xp{1}.core,Xp{1}.U,-n);
                Zp{1} = double(tenmat(Zp{1},n));
                
                Zp{2} = ttm(Xp{2}.core,Xp{2}.U,-n);
                Zp{2} = double(tenmat(Zp{2},n));
                
                
                UnYn = Xp{1}.U{n}'*double(tenmat(Y,n)) - Zp{1};
                s = sum(UnYn(1:R(1,n),:).*Zp{2}(1:R(1,n),:),2)./sum(Zp{2}(1:R(1,n),:).*Zp{2}(1:R(1,n),:),2);
                %s = max(0,min(s,1-1e-8));
                s = min(max(-1,s),1);
                
                
                % Find Q
                [qq,rr] = qr(Xp{1}.U{n});
                ix = find(sum(abs(Xp{1}.U{n}'*qq))<1e-8);
                Unbar = qq(:,ix);
                temp = Unbar'*double(tenmat(Y,n))*Zp{2}'*diag(sqrt(1-s.^2));
                [ut,st,vt] = svd(temp);
                Qn = ut*vt';
                
                Xp{2}.U{n} = Xp{1}.U{n} * diag(s) + Unbar * Qn * diag(sqrt(1-s.^2));
            end
            
            % update core
            if P == 2
                Xp = fastupdatecore(Y,Xp);
            else
                Xp = updatecore(Y,Xp);
            end
            
            % Approximate error
            abserror = Frobnorm(Xp,normY);
            err = abserror/normY;
            %Err = [Err err];
            %fprintf('Err2 %d\n',err)
            if ~isinf(errold2)
                done2 = abs(err-errold2) < errold2 * 1e-4;
            end
            errold2 = err;
            
        end
    end

%% *********************************************************************
    function [Xp,err] = correcttwofactors(Xp,err)
        for n = 1:N
            [u,s,v] = svd(Xp{1}.U{n}'*Xp{2}.U{n},0);
            Xp{1}.U{n} = Xp{1}.U{n} * u;
            Xp{2}.U{n} = Xp{2}.U{n} * v;
            
            Xp{1}.core = ttm(Xp{1}.core,u',n);
            Xp{2}.core = ttm(Xp{2}.core,v',n);
        end
        
        done2 = false;errold2 = errold;cnt = 0;
        while  ~done2 || cnt < 20
            cnt = cnt + 1;
            
            % Find s1 sequentially
            n = 1;
            Zp{1} = ttm(Xp{1}.core,Xp{1}.U,-n);
            Zp{1} = double(tenmat(Zp{1},n));
            
            Zp{2} = ttm(Xp{2}.core,Xp{2}.U,-n);
            Zp{2} = double(tenmat(Zp{2},n));
            
            UnYn = Xp{2}.U{n}'*double(tenmat(Y,n)) - Zp{2};
            s1 = sum(UnYn(1:R(1,n),:).*Zp{1},2)./sum(Zp{1}.*Zp{1},2);
            
            % Find s2 sequentially
            n = 2;
            Zp{1} = ttm(Xp{1}.core,Xp{1}.U,-n);
            Zp{1} = double(tenmat(Zp{1},n));
            
            Zp{2} = ttm(Xp{2}.core,Xp{2}.U,-n);
            Zp{2} = double(tenmat(Zp{2},n));
            
            UnYn = Xp{2}.U{n}'*double(tenmat(Y,n)) - Zp{2};
            s2 = sum(UnYn(1:R(1,n),:).*Zp{1},2)./sum(Zp{1}.*Zp{1},2);
            
            %             % S from current estimate
            %             s1 = diag(Xp{1}.U{1}'*Xp{2}.U{1});
            %             s2 = diag(Xp{1}.U{2}'*Xp{2}.U{2});
            
            s1 = min(max(-1,s1),1);
            s2 = min(max(-1,s2),1);
            
            s1b = sqrt(1-s1.^2);
            s2b = sqrt(1-s2.^2);
            
            % Find Q1
            [qq,rr] = qr(Xp{2}.U{1});
            ix = find(sum(abs(Xp{2}.U{1}'*qq))<1e-8);
            U1bar = qq(:,ix);
            
            [qq,rr] = qr(Xp{2}.U{2});
            ix = find(sum(abs(Xp{2}.U{2}'*qq))<1e-8);
            U2bar = qq(:,ix);
            
            temp = ttm(Y,{U1bar Xp{2}.U{2}},[1 2],'t');
            temp2 = ttm(Xp{1}.core,{diag(s1b), diag(s2)},[1 2]);
            
            temp = ttt(temp,temp2,[2 3]);
            [ut,st,vt] = svd(double(temp));
            Q1 = ut*vt';
            
            % Find Q2
            temp = ttm(Y,{Xp{2}.U{1} U2bar },[1 2],'t');
            temp2 = ttm(Xp{1}.core,{diag(s1), diag(s2b)},[1 2]);
            
            temp = ttt(temp,temp2,[1 3]);
            [ut,st,vt] = svd(double(temp));
            Q2 = ut*vt';
            
            
            Xp{1}.U{1} = Xp{2}.U{1} * diag(s1) + U1bar * Q1 * diag(s1b);
            Xp{1}.U{2} = Xp{2}.U{2} * diag(s2) + U2bar * Q2 * diag(s2b);
            
            % update core
            W =[];
            for p = 1:P
                KRN = kron(Xp{p}.U{N-1},Xp{p}.U{N-2});
                for kn = N-3:-1:1
                    KRN = kron(KRN,Xp{p}.U{kn});
                end
                W = [W KRN];
            end
            G = W\reshape(Y.data,I(1)*I(2),I(3));
            
            G = mat2cell(G,prod(R,2),I(N));
            for p = 1:P
                Xp{p}.core = tensor(reshape(G{p},[R(p,:) I(N)]));
            end
            
            
            %% Correct A2
            % Alternatively Find s1 sequentially
            n = 1;
            Zp{1} = ttm(Xp{1}.core,Xp{1}.U,-n);
            Zp{1} = double(tenmat(Zp{1},n));
            
            Zp{2} = ttm(Xp{2}.core,Xp{2}.U,-n);
            Zp{2} = double(tenmat(Zp{2},n));
            
            UnYn = Xp{1}.U{n}'*double(tenmat(Y,n)) - Zp{1};
            s1 = sum(UnYn(1:R(1,n),:).*Zp{2},2)./sum(Zp{2}.*Zp{2},2);
            
            % Find s2 sequentially
            n = 2;
            Zp{1} = ttm(Xp{1}.core,Xp{1}.U,-n);
            Zp{1} = double(tenmat(Zp{1},n));
            
            Zp{2} = ttm(Xp{2}.core,Xp{2}.U,-n);
            Zp{2} = double(tenmat(Zp{2},n));
            
            UnYn = Xp{1}.U{n}'*double(tenmat(Y,n)) - Zp{1};
            s2 = sum(UnYn(1:R(1,n),:).*Zp{2},2)./sum(Zp{2}.*Zp{2},2);
            
            %%% S from current estimate
            %s1 = diag(Xp{1}.U{1}'*Xp{2}.U{1});
            %s2 = diag(Xp{1}.U{2}'*Xp{2}.U{2});
            
            s1 = min(max(-1,s1),1);
            s2 = min(max(-1,s2),1);
            
            s1b = sqrt(1-s1.^2);
            s2b = sqrt(1-s2.^2);
            
            % Find Q1
            [qq,rr] = qr(Xp{1}.U{1});
            ix = find(sum(abs(Xp{1}.U{1}'*qq))<1e-8);
            U1bar = qq(:,ix);
            
            [qq,rr] = qr(Xp{1}.U{2});
            ix = find(sum(abs(Xp{1}.U{2}'*qq))<1e-8);
            U2bar = qq(:,ix);
            
            temp = ttm(Y,{U1bar Xp{1}.U{2}},[1 2],'t');
            temp2 = ttm(Xp{2}.core,{diag(s1b), diag(s2)},[1 2]);
            
            temp = ttt(temp,temp2,[2 3]);
            [ut,st,vt] = svd(double(temp));
            Q1 = ut*vt';
            
            % Find Q2
            temp = ttm(Y,{Xp{1}.U{1} U2bar },[1 2],'t');
            temp2 = ttm(Xp{2}.core,{diag(s1), diag(s2b)},[1 2]);
            
            temp = ttt(temp,temp2,[1 3]);
            [ut,st,vt] = svd(double(temp));
            Q2 = ut*vt';
            
            Xp{2}.U{1} = Xp{1}.U{1} * diag(s1) + U1bar * Q1 * diag(s1b);
            Xp{2}.U{2} = Xp{1}.U{2} * diag(s2) + U2bar * Q2 * diag(s2b);
            
            % update core
            W =[];
            for p = 1:P
                KRN = kron(Xp{p}.U{N-1},Xp{p}.U{N-2});
                for kn = N-3:-1:1
                    KRN = kron(KRN,Xp{p}.U{kn});
                end
                W = [W KRN];
            end
            G = W\reshape(Y.data,I(1)*I(2),I(3));
            
            G = mat2cell(G,prod(R,2),I(N));
            for p = 1:P
                Xp{p}.core = tensor(reshape(G{p},[R(p,:) I(N)]));
            end
            
            %% Approximation error
            abserror = Frobnorm(Xp,normY);
            err = abserror/normY;
            %Err = [Err err];
            %fprintf('Err2 %d\n',err)
            if ~isinf(errold2)
                done2 = abs(err-errold2) < errold2 * 1e-4;
            end
            errold2 = err;
            
        end
    end

%% *********************************************************************
%     function Xp = als_procrustes(Y,Xp)
%         %         Xp = rotatefactor(Xp);
%         for n2 = 1:N
%             %             if isa(Xp{1},'ttensor')
%             %                 Zp{1} = ttm(Xp{1}.core,Xp{1}.U,-n2);
%             %                 Zp{1} = double(tenmat(Zp{1},n2));
%             %             elseif isa(Xp{1},'ktensor')
%             %                 tt = khatrirao(Xp{1}.U([1:n2-1 n2+1:end]),'r')';
%             %                 Zp{1} = diag(Xp{1}.lambda) * tt;
%             %             end
%             %
%             %
%             %             if isa(Xp{2},'ttensor')
%             %                 Zp{2} = ttm(Xp{2}.core,Xp{2}.U,-n2);
%             %                 Zp{2} = double(tenmat(Zp{2},n2));
%             %             elseif isa(Xp{2},'ktensor')
%             %                 Zp{2} = diag(Xp{2}.lambda) * khatrirao(Xp{2}.U([1:n2-1 n2+1:end]),'r')';
%             %             end
%             %
%             %
%             %             Un = double(tenmat(Y,n2))/[Zp{1}; Zp{2}];
%             %             Un = mat2cell(Un,I(n2),R(:,n2)');
%
%             % Fast update Un
%             utu = cellfun(@(x,y) x'*y,Xp{1}.u,Xp{2}.u,'uni',0);
%             t1 = ttt(Xp{1}.core,Xp{1}.core,[1:n2-1 n2+1:N]);
%             t2 = ttm(Xp{1}.core,utu,-n2,'t');
%             t2 = ttt(tensor(t2),Xp{2}.core,[1:n2-1 n2+1:N]);
%             t3 = ttt(Xp{2}.core,Xp{2}.core,[1:n2-1 n2+1:N]);
%             t4 = [double(t1) double(t2); double(t2)' double(t3)];
%
%             t5 = [double(ttt(ttm(Y,Xp{1}.U,-n2,'t'),Xp{1}.core,[1:n2-1 n2+1:N])), ...
%                 double(ttt(ttm(Y,Xp{2}.U,-n2,'t'),Xp{2}.core,[1:n2-1 n2+1:N]))];
%             Un = t5/t4;
%             Un = mat2cell(Un,I(n2),R(:,n2)');
%
%
%             % Rotate factors to be orthogonal
%             for p2 = 1:P
%                 Xp{p2}.U{n2} = Un{p2};
%                 [Xp{p2}.U{n2},rr] = qr(Xp{p2}.U{n2},0);
%
%                 if isa(Xp{p2},'ttensor')
%                     Xp{p2}.core = ttm(Xp{p2}.core,rr,n2);
%                 elseif isa(Xp{p2},'ktensor')
%                     Xp{p2}.lambda= Xp{p2}.lambda*rr;
%                 end
%             end
%
%             % decorrelate factors among block terms
%             [u,s,v] = svd(Xp{1}.U{n2}'*Xp{2}.U{n2},0);
%             Xp{1}.U{n2} = Xp{1}.U{n2} * u;
%             Xp{2}.U{n2} = Xp{2}.U{n2} * v;
%
%             if isa(Xp{1},'ttensor')
%                 Xp{1}.core = ttm(Xp{1}.core,u',n2);
%             elseif isa(Xp{1},'ktensor')
%                 Xp{1}.lambda = Xp{1}.lambda * u';
%             end
%
%             if isa(Xp{2},'ttensor')
%                 Xp{2}.core = ttm(Xp{2}.core,v',n2);
%             elseif isa(Xp{2},'ktensor')
%                 Xp{2}.lambda = Xp{2}.lambda * v';
%             end
%
%
%
%             %             for k2 = 1:5
%             %
%             %                 % Find s2
%             %                 [U2bar,rr] = qr(Xp{2}.U{n});
%             %                 U2bar(:,1:R(2,n)) = Xp{2}.U{n};
%             %
%             %                 YtAn = tenmat(ttm(Y,U2bar',n),n);
%             %                 temp2 = YtAn(1:R(2,n),:) - Zp{2};
%             %                 s = sum(temp2 .* Zp{1},2)./sum(Zp{1}.* Zp{1},2);
%             %                 s = min(max(-1,s),1);
%             %
%             %                 % Find Q2
%             %                 temp = YtAn(R(2,n)+1:end,:)*Zp{1}'*diag(sqrt(1-s.^2));
%             %                 [ut,st,vt] = svd(temp);
%             %                 Qn = ut*vt';
%             %
%             %                 Xp{1}.U{n} = U2bar * [diag(s); Qn * diag(sqrt(1-s.^2))];
%             %
%             %
%             %                 % Find s1
%             %                 [U1bar,rr] = qr(Xp{1}.U{n});
%             %                 U1bar(:,1:R(1,n)) = Xp{1}.U{n};
%             %
%             %                 YtAn = tenmat(ttm(Y,U1bar',n),n);
%             %                 temp2 = YtAn(1:R(1,n),:) - Zp{1};
%             %                 s = sum(temp2 .* Zp{2},2)./sum(Zp{2}.* Zp{2},2);
%             %                 s = min(max(-1,s),1);
%             %
%             %                 % Find Q1
%             %                 temp = YtAn(R(1,n)+1:end,:)*Zp{2}'*diag(sqrt(1-s.^2));
%             %                 [ut,st,vt] = svd(temp);
%             %                 Qn = ut*vt';
%             %
%             %                 Xp{2}.U{n} = U1bar * [diag(s); Qn * diag(sqrt(1-s.^2))];
%             %             end
%
%         end
%
%         if P == 2
%             Xp = fastupdatecore(Y,Xp);
%         else
%             Xp = updatecore(Y,Xp);
%         end
%
%     end


% *********************************************************************
%     %% Procrustes
%     if param.updatefactor
%         V = cell(P,N);
%         for n = 1:N-1
%             Zp = cell(P,1);
%             for p = 1:P
%                 Zpp = ttm(Xp{p}.core,Xp{p}.U,-n);
%                 Zp{p} = double(tenmat(Zpp,n));
%             end
%             opts = mprocrustes;
%             opts.verbose = 0;opts.normX = normY;opts.tol = 1e-8;
%             opts.init = cellfun(@(x) x.U{n},Xp,'uni',0);
%             Yn = double(tenmat(Y,n));
%             [Uc,Err2] = mprocrustes(Yn,Zp,opts);
%             %             semilogy(Err2)
%
%             Err = [Err Err2];
%             for p = 1:P
%                 Xp{p}.U{n} = Uc{p};
%             end
%             %V(:,n) = Uc;
%
%             %% Estimate Gp
%             W =[];
%             for p = 1:P
%                 KRN = kron(Xp{p}.U{N-1},Xp{p}.U{N-2});
%                 for kn = N-3:-1:1
%                     KRN = kron(KRN,Xp{p}.U{kn});
%                 end
%                 W = [W KRN];
%             end
%             G = W\reshape(Y.data,I(1)*I(2),I(3));
%
%             %Fast update
%
%             %             for p = 1:P
%             %                 T = ttm(Y,Xp{p}.U,'t');
%             %                 YtU(sum(sta(1:p))+1:sum(sta(1:p+1))) =  T.data(:);
%             %             end
%             %
%             %
%             %             for p1 = 1:P-1
%             %                 for p2 = p1+1:P
%             %                     KR = Xp{p1}.U{end}' * Xp{p2}.U{end};
%             %                     for n = N-1:-1:1
%             %                         KR = kron(KR,Xp{p1}.U{n}' * Xp{p2}.U{n});
%             %                     end
%             %                     str = sum(sta(1:p1))+1;
%             %                     stc = sum(sta(1:p2))+1;
%             %                     WtW(str:str+sta(p1+1)-1,stc:stc+sta(p2+1)-1) = KR;
%             %                     WtW(stc:stc+sta(p2+1)-1,str:str+sta(p1+1)-1) = KR';
%             %                 end
%             %             end
%             %             G = WtW\YtU;
%             % fast inverse of WtW
%             %  a = 1./(1-s.^2);
%             %  b = -s./(1-s.^2);
%
%             G = mat2cell(G,prod(R,2),I(N));
%             for p = 1:P
%                 Xp{p}.core = tensor(reshape(G{p},[R(p,:) I(N)]));
%             end
%
%         end
%     end



%% ********************************************************************
%     function Xp= corr_deg(Xp)
%         % Call after rotating factors
%         for n2 = 1:N
%             cdeg(n2,:) = diag(Xp{1}.U{n2}'*Xp{2}.U{n2});
%
%             ix = find(cdeg(n2,:) > 0.99);
%             if ~isempty(ix)
%                 cdeg(n2,ix) = cdeg(n2,ix)*.3;
%                 %                 [u,s,v] = svd(Xp{1}.U{n2}'*Xp{2}.U{n2},0);
%                 [Unbar,rr] = qr(Xp{2}.U{n2});
%
%                 % Find Q
%                 Z = ttm(Xp{1}.core,Xp{1}.U,-n2);
%                 Z = double(tenmat(Z,n2));
%                 temp = Unbar(:,size(Xp{2}.U{n2},2)+1:end)'*...
%                     double(tenmat(Y,n2))*Z'*diag(sqrt(1-cdeg(n2,:).^2));
%                 [ut,st,vt] = svd(temp);
%                 Qn = ut*vt' * diag(sqrt(1-cdeg(n2,:).^2));
%
%                 Xp{1}.U{n2} = Xp{2}.U{n2} * diag(cdeg(n2,:)) + Unbar(:,size(Xp{2}.U{n2},2)+1:end)* Qn;
%             end
%
%         end
%
%         %         [ix,ix2] = find(cdeg > 0.99);
%         %         for ki = 1:numel(ix)
%         %             if ix(ki) == 1
%         %                 [A B1 S]=btd_ajd(Y.data,R(:,1));
%         %             end
%         %             if ix(ki) == 2
%         %                 [B A1 S]=btd_ajd(permute(Y.data,[2 1 3]),R(:,1));
%         %             end
%         %             if ix(ki) == 3
%         %                 [C A1 S]=btd_ajd(permute(Y.data,[3 1 2]),R(:,1));
%         %             end
%         %         end
%     end


%% *********************************************************************
    function Xp = updateAG(Y,Xp)
        % Fast ALS update simultanously one factor and core tensor
        
        %         utu = cellfun(@(x,y) y'*x,Xp{1}.u,Xp{2}.u,'uni',0);
        utu = cellfun(@(x,y) diag(y'*x),Xp{1}.u,Xp{2}.u,'uni',0);
        for n2 = 1:N
            % Fast update Un
            T1 = ttm(Y,Xp{1}.U,-n2,'t');
            T2 = ttm(Y,Xp{2}.U,-n2,'t');
            T1 = double(tenmat(T1,n2));
            T2 = double(tenmat(T2,n2));
            
            W = full(ktensor(utu([1:n2-1 n2+1:N])));
            
            F1 = ((T1 - T2 * diag(W(:)))* diag(1./(1-W(:).^2)));
            F2 = ((T2 - T1 * diag(W(:)))* diag(1./(1-W(:).^2)));
            
            
            G1 = double(tenmat(Xp{1}.core,n2));
            errA = [];
            for k3 = 1:100
                A = F1/G1;
                A = bsxfun(@rdivide,A,sqrt(sum(A.^2)));
                G1 = A\F1;
                errA(k3) = norm(F1 - A*G1,'fro');
                if (k3 > 1) && abs(errA(k3)-errA(k3-1))< errA(k3) * 1e-4
                    break
                end
            end
            
            
            G2 = double(tenmat(Xp{2}.core,n2));
            errA = [];
            for k3 = 1:100
                B = F2/G2;
                B = bsxfun(@rdivide,B,sqrt(sum(B.^2)));
                G2 = B\F2;
                errA(k3) = norm(F2 - B*G2,'fro');
                if (k3 > 1) && abs(errA(k3)-errA(k3-1))< errA(k3) * 1e-4
                    break
                end
            end
            
            
            Xp{1}.U{n2} = A;
            Xp{1}.core = tensor(itenmat(G1,n2,R(1,:)));
            Xp{2}.U{n2} = B;
            Xp{2}.core = tensor(itenmat(G2,n2,R(2,:)));
            
            % Rotate factors to be orthogonal
            for p2 = 1:P
                [Xp{p2}.U{n2},rr] = qr(Xp{p2}.U{n2},0);
                
                if isa(Xp{p2},'ttensor')
                    Xp{p2}.core = ttm(Xp{p2}.core,rr,n2);
                elseif isa(Xp{p2},'ktensor')
                    Xp{p2}.lambda= Xp{p2}.lambda*rr;
                end
            end
            
            % decorrelate factors among block terms
            [u,s,v] = svd(Xp{1}.U{n2}'*Xp{2}.U{n2},0);
            Xp{1}.U{n2} = Xp{1}.U{n2} * u;
            Xp{2}.U{n2} = Xp{2}.U{n2} * v;
            
            %utu{n2} = s';
            utu{n2} = diag(s');
            
            if isa(Xp{1},'ttensor')
                Xp{1}.core = ttm(Xp{1}.core,u',n2);
            elseif isa(Xp{1},'ktensor')
                Xp{1}.lambda = Xp{1}.lambda * u';
            end
            
            if isa(Xp{2},'ttensor')
                Xp{2}.core = ttm(Xp{2}.core,v',n2);
            elseif isa(Xp{2},'ktensor')
                Xp{2}.lambda = Xp{2}.lambda * v';
            end
            
            Xp = fastupdatecore(Y,Xp);
        end
        %         Xp = rotatefactor(Xp);
        %         Xp = fastupdatecore(Y,Xp);
    end


%% *********************************************************************
    function Xp = updateAG_svd0(Y,Xp)
        % Fast ALS update simultanously one factor and core tensor
        
        %         utu = cellfun(@(x,y) y'*x,Xp{1}.u,Xp{2}.u,'uni',0);
        utu = cellfun(@(x,y) diag(y'*x),Xp{1}.u,Xp{2}.u,'uni',0);
        for n2 = 1:N
            % Fast update Un
            T1 = ttm(Y,Xp{1}.U,-n2,'t');
            T2 = ttm(Y,Xp{2}.U,-n2,'t');
            T1 = double(tenmat(T1,n2));
            T2 = double(tenmat(T2,n2));
            
            W = full(ktensor(utu([1:n2-1 n2+1:N])));
            
            F1 = ((T1 - T2 * diag(W(:)))* diag(1./(1-W(:).^2)));
            [A,s,G1] = svds(F1,R(1,n2));
            G1 = s*G1';
            
            
            F2 = ((T2 - T1 * diag(W(:)))* diag(1./(1-W(:).^2)));
            [B,s,G2] = svds(F2,R(2,n2));
            G2 = s*G2';
            
            
            Xp{1}.U{n2} = A;
            Xp{1}.core = tensor(itenmat(G1,n2,R(1,:)));
            Xp{2}.U{n2} = B;
            Xp{2}.core = tensor(itenmat(G2,n2,R(2,:)));
            
            %             % Rotate factors to be orthogonal
            %             for p2 = 1:P
            %                 [Xp{p2}.U{n2},rr] = qr(Xp{p2}.U{n2},0);
            %
            %                 if isa(Xp{p2},'ttensor')
            %                     Xp{p2}.core = ttm(Xp{p2}.core,rr,n2);
            %                 elseif isa(Xp{p2},'ktensor')
            %                     Xp{p2}.lambda= Xp{p2}.lambda*rr;
            %                 end
            %             end
            
            % decorrelate factors among block terms
            [u,s,v] = svd(Xp{1}.U{n2}'*Xp{2}.U{n2},0);
            Xp{1}.U{n2} = Xp{1}.U{n2} * u;
            Xp{2}.U{n2} = Xp{2}.U{n2} * v;
            
            %utu{n2} = s';
            utu{n2} = diag(s');
            
            if isa(Xp{1},'ttensor')
                Xp{1}.core = ttm(Xp{1}.core,u',n2);
            elseif isa(Xp{1},'ktensor')
                Xp{1}.lambda = Xp{1}.lambda * u';
            end
            
            if isa(Xp{2},'ttensor')
                Xp{2}.core = ttm(Xp{2}.core,v',n2);
            elseif isa(Xp{2},'ktensor')
                Xp{2}.lambda = Xp{2}.lambda * v';
            end
            
            Xp = fastupdatecore(Y,Xp);
        end
        %         Xp = rotatefactor(Xp);
        %         Xp = fastupdatecore(Y,Xp);
    end


%% *********************************************************************
    function Xp = updateAG_svd(Y,Xp)
        % Fast ALS update simultanously one factor and core tensor
        utu = cellfun(@(x,y) diag(y'*x),Xp{1}.u,Xp{2}.u,'uni',0);
        for n2 = 1:N
            % Fast update Un
            T1 = ttm(Y,Xp{1}.U,-n2,'t');
            T2 = ttm(Y,Xp{2}.U,-n2,'t');
            
            T1 = permute(T1,[1:n2-1 n2+1:N n2]);
            T2 = permute(T2,[1:n2-1 n2+1:N n2]);
            T1 = double(T1);
            T2 = double(T2);
            
            W = double(full(ktensor(utu([1:n2-1 n2+1:N]))));
            F1 = T2(1:R(1,1),1:R(1,2),:);
            F1 = bsxfun(@times,F1,W);
            F1 = T1 - F1;
            F1 = bsxfun(@rdivide,F1,1-W.^2);
            
            F1 = reshape(F1,[],size(Y,n2))';
            [A,s,G1] = svds(F1,R(1,n2));
            G1 = s*G1';
            G1 = itenmat(G1,n2,R(1,:));
            
            
            F2 = T2;
            T1b = bsxfun(@times,T1,W);
            T1b = F2(1:R(1,1),1:R(1,1),:) - T1b;
            T1b = bsxfun(@rdivide,T1b,1-W.^2);
            F2(1:R(1,1),1:R(1,1),:) = T1b;
            F2 = reshape(F2,[],size(Y,n2))';
            
            [B,s,G2] = svds(F2,R(2,n2));
            G2 = s*G2';
            G2 = itenmat(G2,n2,R(2,:));
            %             G2 = permute(G2,[1:n2-1 n2+1:N n2]);
            %             G2 = bsxfun(@rdivide,G2,1-W.^2);
            
            
            Xp{1}.U{n2} = A;
            Xp{1}.core = tensor(G1);
            Xp{2}.U{n2} = B;
            Xp{2}.core = tensor(G2);
            
            % Rotate factors to be orthogonal
            
            % decorrelate factors among block terms
            [u,s,v] = svd(Xp{1}.U{n2}'*Xp{2}.U{n2},0);
            Xp{1}.U{n2} = Xp{1}.U{n2} * u;
            Xp{2}.U{n2} = Xp{2}.U{n2} * v;
            
            %utu{n2} = s';
            utu{n2} = diag(s');
            
            if isa(Xp{1},'ttensor')
                Xp{1}.core = ttm(Xp{1}.core,u',n2);
            elseif isa(Xp{1},'ktensor')
                Xp{1}.lambda = Xp{1}.lambda * u';
            end
            
            if isa(Xp{2},'ttensor')
                Xp{2}.core = ttm(Xp{2}.core,v',n2);
            elseif isa(Xp{2},'ktensor')
                Xp{2}.lambda = Xp{2}.lambda * v';
            end
        end
        %         Xp = rotatefactor(Xp);
        Xp = fastupdatecore(Y,Xp);
    end

%%*******
%% *********************************************************************
    function Xp = updateAG_svd2(Y,Xp)
        % Fast ALS update simultanously one factor and core tensor
        utu = cellfun(@(x,y) diag(y'*x),Xp{1}.u,Xp{2}.u,'uni',0);
        for n2 = 1:N
            % Fast update Un
            T1 = ttm(Y,Xp{1}.U,-n2,'t');
            T2 = ttm(Y,Xp{2}.U,-n2,'t');
            
            T1 = permute(T1,[1:n2-1 n2+1:N n2]);
            T2 = permute(T2,[1:n2-1 n2+1:N n2]);
            T1 = double(T1);
            T2 = double(T2);
            
            W = double(full(ktensor(utu([1:n2-1 n2+1:N]))));
            F1 = T2(1:R(1,1),1:R(1,2),:);
            F1 = bsxfun(@times,F1,W);
            F1 = T1 - F1;
            %F1 = bsxfun(@rdivide,F1,1-W.^2);
            
            F1 = reshape(F1,[],size(Y,n2))';
            [A,s,G1] = svds(F1,R(1,n2));
            G1 = s*G1';
            G1 = bsxfun(@rdivide,G1,1-W(:)'.^2);
            G1 = itenmat(G1,n2,R(1,:));
            
            
            F2 = T2;
            T1b = bsxfun(@times,T1,W);
            T1b = F2(1:R(1,1),1:R(1,1),:) - T1b;
            %T1b = bsxfun(@rdivide,T1b,1-W.^2);
            F2(1:R(1,1),1:R(1,1),:) = T1b;
            F2 = reshape(F2,[],size(Y,n2))';
            
            [B,s,G2] = svds(F2,R(2,n2));
            G2 = s*G2';
            G2 = itenmat(G2,n2,R(2,:));
            G2 = permute(G2,[1:n2-1 n2+1:N n2]);
            G2(1:R(1,1),1:R(1,1),:) = bsxfun(@rdivide,G2(1:R(1,1),1:R(1,1),:),1-W.^2);
            G2 = ipermute(G2,[1:n2-1 n2+1:N n2]);
            
            Xp{1}.U{n2} = A;
            Xp{1}.core = tensor(G1);
            Xp{2}.U{n2} = B;
            Xp{2}.core = tensor(G2);
            
            % Rotate factors to be orthogonal
            
            % decorrelate factors among block terms
            [u,s,v] = svd(Xp{1}.U{n2}'*Xp{2}.U{n2},0);
            Xp{1}.U{n2} = Xp{1}.U{n2} * u;
            Xp{2}.U{n2} = Xp{2}.U{n2} * v;
            
            %utu{n2} = s';
            utu{n2} = diag(s');
            
            if isa(Xp{1},'ttensor')
                Xp{1}.core = ttm(Xp{1}.core,u',n2);
            elseif isa(Xp{1},'ktensor')
                Xp{1}.lambda = Xp{1}.lambda * u';
            end
            
            if isa(Xp{2},'ttensor')
                Xp{2}.core = ttm(Xp{2}.core,v',n2);
            elseif isa(Xp{2},'ktensor')
                Xp{2}.lambda = Xp{2}.lambda * v';
            end
        end
        %         Xp = rotatefactor(Xp);
        Xp = fastupdatecore(Y,Xp);
    end

%% ********************************************************************
    function Xp = bcd_init
        opt_hooi = struct('printitn',0,'init','nvecs','dimorder',1:N);
        
        if strcmp(param.init,'block')
            Xp = cell(P,1);
            Xp{1} = mtucker_als(Y,R(1,:),opt_hooi);
            for p = 2:P
                Yr = Y - tensor(Xp{1});
                Xp{p} = mtucker_als(Yr,R(p,:),opt_hooi);
                % Xp{p}.core =Xp{p}.core;
            end
            
        elseif strcmp(param.init,'commdec')
            
            Yc = squeeze(num2cell(double(Y), [1 2]));
            Yc1 = cellfun(@(x) x*x',Yc,'uni',0);
            [A, e] = commdec(Yc1);
            
            Yc1 = cellfun(@(x) x'*x,Yc,'uni',0);
            [B, e] = commdec(Yc1);
            
            Yc = squeeze(num2cell(double(Y), [1 3]));
            Yc1 = cellfun(@(x) squeeze(x)'*squeeze(x),Yc,'uni',0);
            [C, e] = commdec(Yc1);
            
        elseif strcmp(param.init,'ajd')
            % init using joint diagonal block term
            if any(I>sum(R))
                Ycpr = mtucker_als(Y,sum(R),opt_hooi);
                [A B S]=btd_ajd(Ycpr.core.data,R(:,1));
                
                
                S2=permute(S,[3,2,1]);
                
                Bb = eye(size(S,3));
                [m m1 p]=size(S2);
                S=zeros(m,m,m);
                for n=1:m
                    U=zeros(m,m);
                    for i=1:p
                        u=S2(:,:,i)*Bb(n,:)';
                        U=U+u*u';
                    end
                    S(:,:,n)=U;
                end
                V=sum(S,3);
                for n=1:m
                    [W Lam]=eig(S(:,:,n),V-S(:,:,n));
                    [Lmin,imin]=max(diag(Lam));
                    %    A(j,:)=A(j,:)/norm(A(j,:))
                    C(n,:)=W(:,imin)';
                    C(n,:)=C(n,:)/norm(C(n,:));
                end
                
                %%
                S = ttm(Ycpr.core,{A B C}); % this computation is notimportant since core tensors are recomputed later
                
                A=inv(A);A = Ycpr.U{1}*A;
                B=inv(B);B = Ycpr.U{2}*B;
                C=inv(C);C = Ycpr.U{3}*C;
                
                A = mat2cell(A,size(A,1),R(:,1));
                B = mat2cell(B,size(B,1),R(:,2));
                C = mat2cell(C,size(C,1),R(:,3));
                
                if R(1,1) == 1
                    Xp{1} = ktensor(S(1:R(1,1),1:R(1,2),1:R(1,3)),{A{1},B{1} C{1}});
                else
                    Xp{1} = ttensor(tensor(S(1:R(1,1),1:R(1,2),1:R(1,3))),{A{1},B{1} C{1}});
                end
                if R(2,1) == 1
                    Xp{2} = ktensor(S(R(1,1)+1:end,R(1,2)+1:end,R(1,3)+1:end),...
                        {A{2},B{2} C{2}});
                else
                    Xp{2} = ttensor(tensor(S(R(1,1)+1:end,R(1,2)+1:end,R(1,3)+1:end)),...
                        {A{2},B{2} C{2}});
                end
            else
                [A B S]=btd_ajd(Y.data,R(:,1));
                [C A1 S]=btd_ajd(permute(Y.data,[3 1 2]),R(:,1));
                
                %         [A B S]=btd_ajd2(Y,R(:,1));
                %         [C A1 S]=btd_ajd2(permute(Y,[3 1 2]),R(:,1));
                
                %         S = ttm(Y,{A B},[1 2]);
                %         [C A1 S]=btd_ajd2(permute(S,[3 1 2]),R(:,1));
                %         A = A1*A;
                %
                %         S = ttm(Y,{A C},[1 3]);
                %         [B A1 S]=btd_ajd2(permute(S,[2 1 3]),R(:,1));
                %         A = A1*A;
                
                
                
                %         [A B S]=btd_ajd2(Y,R(:,1));
                %         [A1 C S]=btd_ajd2(permute(Y,[1 3 2]),R(:,1));
                
                %         S = double(S);
                %         S2=permute(S,[3,2,1]);
                %
                %         Bb = eye(I(3));
                %         [m m1 p]=size(S2);
                %         S=zeros(m,m,m);
                %         for n=1:m
                %             U=zeros(m,m);
                %             for i=1:p
                %                 u=S2(:,:,i)*Bb(n,:)';
                %                 U=U+u*u';
                %             end
                %             S(:,:,n)=U;
                %         end
                %         V=sum(S,3);
                %         for n=1:m
                %             [W Lam]=eig(S(:,:,n),V-S(:,:,n));
                %             [Lmin,imin]=max(diag(Lam));
                %             %    A(j,:)=A(j,:)/norm(A(j,:))
                %             C(n,:)=W(:,imin)';
                %             C(n,:)=C(n,:)/norm(C(n,:));
                %         end
                % %
                %%
                S = ttm(Y,{A B C}); % this computation is notimportant since core tensors are recomputed later
                
                Ai=inv(A);%A1= inv(A1); A = A1;
                Bi=inv(B);
                Ci=inv(C);
                
                Ai = mat2cell(Ai,size(Ai,1),R(:,1));
                Bi = mat2cell(Bi,size(Bi,1),R(:,2));
                Ci = mat2cell(Ci,size(Ci,1),R(:,3));
                
                cR = cumsum([ zeros(1,N);R],1);
                for p = 1:P
                    
                    %             if R(1,1) == 1
                    %                 Xp{1} = ktensor((S(1:R(1,1),1:R(1,2),1:R(1,3))),{Ai{1},Bi{1} Ci{1}});
                    %             else
                    %                 Xp{1} = ttensor(tensor(S(1:R(1,1),1:R(1,2),1:R(1,3))),{Ai{1},Bi{1} Ci{1}});
                    %             end
                    if R(p,1) == 1
                        Xp{p} = ktensor(S(cR(p,1)+1:cR(p+1,1),cR(p,2)+1:cR(p+1,2),cR(p,3)+1:cR(p+1,3)),...
                            {Ai{p},Bi{p} Ci{p}});
                    else
                        Xp{p} = ttensor(tensor(S(cR(p,1)+1:cR(p+1,1),cR(p,2)+1:cR(p+1,2),cR(p,3)+1:cR(p+1,3))),...
                            {Ai{p},Bi{p} Ci{p}});
                    end
                end
                
            end
            
        elseif strcmp(param.init,'td')
            if any(I>sum(R))
                Ycpr = mtucker_als(Y,sum(R),opt_hooi);
                
                [A0 B0 C0 S iter]=btd3(double(Ycpr.core),R(:,1));
                
                %                 [A0 B0 C0 S iter]=tedia3b(double(Ycpr.core),20);
            else
                %                 [A0 B0 C0 S iter]=tedia3b(double(Y),20);
                [A0 B0 C0 S iter]=btd3(double(Y),R(:,1));
                
            end
            
            A0 = inv(A0);
            B0 = inv(B0);
            C0 = inv(C0);
            
            %             % Sort components of A and B
            %             SS=sum(abs(S),3);
            %             [u,s,v] = svds(SS - mean(SS(:)),1);
            %             [u,ord1] = sort(u,'descend');
            %             [v,ord2] = sort(v,'descend');
            %             S = S(ord1,ord2,:);
            %             A0 = A0(:,ord1);
            %             B0 = B0(:,ord2);
            %
            %             SS = squeeze(sum(abs(S),2));
            %             [u,s,v] = svds(SS - mean(SS(:)),1);
            %             %[u,ord1] = sort(u,'descend');
            %             [v,ord2] = sort(v,'descend');
            %             S = S(:,:,ord2);
            %             C0 = C0(:,ord2);
            
            if any(I>sum(R))
                A0= Ycpr.U{1}*A0;
                B0= Ycpr.U{2}*B0;
                C0= Ycpr.U{3}*C0;
            end
            Ac = mat2cell(A0,I(1),R(:,1));
            Bc = mat2cell(B0,I(2),R(:,2));
            Cc = mat2cell(C0,I(3),R(:,3));
            
            cR = cumsum([ zeros(1,N);R],1);
            for p = 1:P
                if R(p,1) == 1
                    Xp{p} = ktensor(S(cR(p,1)+1:cR(p+1,1),cR(p,2)+1:cR(p+1,2),cR(p,3)+1:cR(p+1,3)),...
                        {Ac{p},Bc{p} Cc{p}});
                else
                    Xp{p} = ttensor(tensor(S(cR(p,1)+1:cR(p+1,1),cR(p,2)+1:cR(p+1,2),cR(p,3)+1:cR(p+1,3))),...
                        {Ac{p},Bc{p} Cc{p}});
                end
            end
            
            
        elseif strcmp(param.init,'tdWN')
            if any(I>sum(R))
                Ycpr = mtucker_als(Y,sum(R),opt_hooi);
                mxR = max(sum(R));
                if mxR~=min(sum(R))
                    Yc = double(Ycpr.core(:));
                    Yc(mxR^N) = 0;
                    Yc = reshape(Yc,mxR*ones(1,N));
                else
                    Yc = double(Ycpr.core);
                end
                [A0 B0 C0 S iter]=tediaWN(Yc,100,[],'normal');
                
                if mxR~=min(sum(R))
                    A0 = A0(1:sum(R(:,1)),1:sum(R(:,1)));
                    B0 = B0(1:sum(R(:,2)),1:sum(R(:,2)));
                    C0 = C0(1:sum(R(:,3)),1:sum(R(:,3)));
                end
                A0= Ycpr.U{1}*A0;
                B0= Ycpr.U{2}*B0;
                C0= Ycpr.U{3}*C0;
            else
                [A0 B0 C0 S iter]=tediaWN(double(Y),100,[],'normal');
            end
            
            % tdWN works only for the case of cubic blocks, i.e. R1=R2=R3
            [Rs,isr] = sort(sum(R,2),'descend');
            Ac = mat2cell(A0,I(1),R(isr,1));
            Bc = mat2cell(B0,I(2),R(isr,2));
            Cc = mat2cell(C0,I(3),R(isr,3));
            
            cR = cumsum([ zeros(1,N);R(isr,:)],1);
            for p = 1:P
                if Rs(p,1) == N
                    Xp{p} = ktensor(S(cR(p,1)+1:cR(p+1,1),cR(p,2)+1:cR(p+1,2),cR(p,3)+1:cR(p+1,3)),...
                        {Ac{p},Bc{p} Cc{p}});
                else
                    Xp{p} = ttensor(tensor(S(cR(p,1)+1:cR(p+1,1),cR(p,2)+1:cR(p+1,2),cR(p,3)+1:cR(p+1,3))),...
                        {Ac{p},Bc{p} Cc{p}});
                end
            end
            [foe,isr2] = sort(isr,'ascend');
            Xp = Xp(isr2);
            
            
        elseif strcmp(param.init,'dtld')
            Uinit = cp_init(Y,sum(R(:,1)),param);
            Ui = cellfun(@(x) pinv(x) ,Uinit,'uni',0);
            
            S = ttm(Y,Ui); % this computation is notimportant since core tensors are recomputed later
            S= double(S);
            
            A = Uinit{1};B = Uinit{2};C = Uinit{3};

                        
%             [foe,ix] = max(abs(S(:)));
%             ix = ind2sub_full(size(S),ix);
%             A(:,[ix(1) 1]) = A(:,[1 ix(1)]);S([ix(1) 1],:,:) = S([1 ix(1)],:,:);
%             B(:,[ix(2) 1]) = B(:,[1 ix(2)]);S(:,[ix(2) 1],:,:) = S(:,[1 ix(2)],:);
%             C(:,[ix(3) 1]) = C(:,[1 ix(3)]);S(:,:,[ix(3) 1]) = S(:,:,[1 ix(3)]);
%            
%             % Sort components of A and B
%             SS=sum(abs(S),3);
%             [u,s,v] = svds((SS - mean(SS(:))),1);
%             [u,ord1] = sort(u,'descend');
%             [v,ord2] = sort(v,'descend');
%             S = S(ord1,ord2,:);
%             A = A(:,ord1);
%             B = B(:,ord2);
%             
%             SS = squeeze(sum(abs(S),2));
%             [u,s,v] = svds((SS - mean(SS(:))),1);
%             %[u,ord1] = sort(u,'descend');
%             [v,ord2] = sort(v,'descend');
%             S = S(:,:,ord2);
%             C = C(:,ord2);
            
            [foe,ix] = sort(squeeze(abs(S(1:size(S,1)^2+1:end))),'descend'); 
            A = A(:,ix); S = S(ix,:,:);
            B = B(:,ix); S = S(:,ix,:);
            C = C(:,ix); S = S(:,:,ix);


%             [foe,ix] = sort(squeeze(abs(S(1:size(S,1)^2+1:end))));ix = ix(1);
%             %[foe,ix] = max(squeeze(abs(S(1:size(S,1)^2+1:end))));
%             if ix ~=1
%                 A(:,[ix 1]) = A(:,[1 ix]);S([ix 1],:,:) = S([1 ix],:,:);
%                 B(:,[ix 1]) = B(:,[1 ix]);S(:,[ix 1],:,:) = S(:,[1 ix],:);
%                 C(:,[ix 1]) = C(:,[1 ix]);S(:,:,[ix 1]) = S(:,:,[1 ix]);
%             end
            
            Ai = mat2cell(A,size(Uinit{1},1),R(:,1));
            Bi = mat2cell(B,size(Uinit{2},1),R(:,2));
            Ci = mat2cell(C,size(Uinit{3},1),R(:,3));
            
            cR = cumsum([ zeros(1,N);R],1);
            for p = 1:P
                if R(p,1) == 1
                    Xp{p} = ktensor(S(cR(p,1)+1:cR(p+1,1),cR(p,2)+1:cR(p+1,2),cR(p,3)+1:cR(p+1,3))+eps,...
                        {Ai{p},Bi{p} Ci{p}});
                else
                    Xp{p} = ttensor(tensor(S(cR(p,1)+1:cR(p+1,1),cR(p,2)+1:cR(p+1,2),cR(p,3)+1:cR(p+1,3))),...
                        {Ai{p},Bi{p} Ci{p}});
                end
            end
            
        elseif iscell(param.init)
            Xp = param.init;
        end
        
        % % Rotate factors
        Xp = rotatefactor(Xp);
        
        %         if P == 2
        
        if N == 3
%             ah = Xp{1}.u; Uh = Xp{2}.u;
%             u = cellfun(@(x) x(:,1),Uh,'uni',0);
%             rho = real(cellfun(@(x,y) x'*y(:,1),ah,Uh));
%             rhoall = prod(rho);
%             ldah = (ttv(Y,ah) - rhoall*ttv(Y,u))/(1-rhoall^2);
%             
%             G = ttm(Y,Uh,'t');G = G.data;
%             G(1) = G(1) - ldah *rhoall;
%             G = tensor(G);
%             
%             Xp{1} = ktensor(ldah,ah);
%             if size(Uh{1},2)> 1
%                 Xp{2} = ttensor(G,Uh);
%             else
%                 Xp{2} = ktensor(double(G),Uh);
%             end
            Xp = fastupdatecore(Y,Xp);
        elseif N == 4
            Xp = fastupdatecore_4way(Y,Xp);
        end
        %         else
        %             Xp = updatecore(Y,Xp);
        %         end
        
    end

%%
    function Xp = als_rank1R_v2(Y,Xp) 
        % This algorithm works for both real-valued and complex-valued tensors
        % This ALS algorithm is used to update factor matrices when the
        % first block Xp{1} is of rank-1.
        %         Xp = rotatefactor(Xp);
        
        rho = cellfun(@(x,y) real(x'*y(:,1)),Xp{1}.U,Xp{2}.U);
        for n2 = 1:N
            %% CORRECT FORM %% Fast update when block 1 has rank of R1 = 1
            %lda = Xp{1}.lambda;
            if isa(Xp{2},'ttensor')
                Gn = double(tenmat(Xp{2}.core,n2));
            else
                Gn = Xp{2}.lambda;
            end
            
            %q = abs(prod(cellfun(@(x,y) x'*y(:,1),Xp{1}.U([1:n2-1 n2+1:N]),Xp{2}.U([1:n2-1 n2+1:N]))));
            q = prod(rho([1:n2-1 n2+1:N]));
            
            T1 = double(ttv(Y,Xp{1}.U,[1:n2-1 n2+1:N]));
            T2 = ttm(Y,Xp{2}.U,[1:n2-1 n2+1:N],'t');
            T2 = double(tenmat(T2,n2))*Gn';
            
            
%             F = [abs(Xp{1}.lambda)^2  Xp{1}.lambda*q*Gn(:,1)';
%                 q*conj(Xp{1}.lambda)*Gn(:,1)  Gn*Gn'];  
%             aV = [Xp{1}.lambda'*T1 T2]/(F);an = aV(:,1); Vn = aV(:,2:end);


            Cg = Gn*Gn';
            
            % Closed-form for an and Un    
            Vn = (T2 - q*T1*Gn(:,1)')/(Cg - q^2 * Gn(:,1)*Gn(:,1)' + 1e-8*eye(size(Cg)));
            an = (T1 - q*Vn*Gn(:,1))/Xp{1}.lambda;
            

%             % Closed-form with regularization to minimize rho_n 12/03/2014
%             % mu is regularized parameter
%              
%             mu = 0;
%             ge = Gn(:,1); ge(1) = ge(1) + mu/Xp{1}.lambda;
%             Vn = (T2 - q*T1 * ge')/(Cg - q^2 * (ge*ge'));
%             an = (T1 - q*Vn*ge)/Xp{1}.lambda; 
            
            % Block 1
            ell1 = norm(an);
            Xp{1}.U{n2} = an/ell1;
            Xp{1}.lambda = Xp{1}.lambda * ell1;
            
            % decorrelate factors among block terms
            [Vn,rU] = qr(Vn,0);
            [v,rv] = qr(Vn'*Xp{1}.U{n2});
            
            Xp{2}.U{n2} = Vn * v;
            if isa(Xp{2},'ttensor')
                Xp{2}.core = ttm(Xp{2}.core,v'*rU,n2);
            elseif isa(Xp{2},'ktensor')
                Xp{2}.lambda = Xp{2}.lambda * v'*rU;
            end
            
            rho(n2) = real(Xp{1}.U{n2}'*Xp{2}.U{n2}(:,1));
        end
        
        
%         % Correct size and rank of the second block
%         for n2 = 1:N
%             %% CORRECT FORM %% Fast update when block 1 has rank of R1 = 1
%             %lda = Xp{1}.lambda;
%             if isa(Xp{2},'ttensor')
%                 Gn = double(tenmat(Xp{2}.core,n2));
%                 if rank(Gn) < size(Gn,1)
%                     Cg = Gn*Gn';
%                     [ug,sg] = eig(Cg,0);
%                     Xp{2}.core = ttm(Xp{2}.core,ug');
%                     Xp{2}.u{n2} = Xp{2}.u{n2}*ug;
%                 end
%             end
%         end



        if N == 3
%             Uh = Xp{2}.u;ah = Xp{1}.u;
%             u = cellfun(@(x) x(:,1),Uh,'uni',0);
%             rho = real(cellfun(@(x,y) x'*y(:,1),ah,Uh));
%             rhoall = prod(rho);
%             ldah = (ttv(Y,ah) - rhoall*ttv(Y,u))/(1-rhoall^2);
% 
%             G = ttm(Y,Uh,'t');G = G.data;
%             G(1) = G(1) - ldah *rhoall;
%             G = tensor(G);
% 
%             Xp{1} = ktensor(ldah,ah);
%             if size(Uh{1},2)> 1
%                 Xp{2} = ttensor(G,Uh);
%             else
%                 Xp{2} = ktensor(double(G),Uh);
%             end

            Xp= fastupdatecore(Y,Xp);
        elseif N == 4
            Xp= fastupdatecore_4way(Y,Xp);
        end
        
%         eval_fastcost = false;
        %Xp= fastupdatecore_maxblock1(Y,Xp);
%         Xp= fastupdatecore_minblock1(Y,Xp);
        %Xp= fastupdatecore_maxblock2(Y,Xp);
    end

    function Xp = fastupdatecore_maxblock1(Y,Xp)
        % Update core tensors of block terms
        % Call after rotate factors to be independent
        % update core
        %G = W\reshape(Y.data,I(1)*I(2),I(3));
        
        UtV = {sum(Xp{1}.U{1}.*Xp{2}.U{1}(:,1:R(1,1)))' ...
            sum(Xp{1}.U{2}.*Xp{2}.U{2}(:,1:R(1,2)))' ...
            sum(Xp{1}.U{3}.*Xp{2}.U{3}(:,1:R(1,3)))'};
        
        G1 = (ttm(Y,Xp{1}.U,'t'));
        G2 = (ttm(Y,Xp{2}.U,'t'));
        
        G3 = (full(ktensor(UtV)));
        
        mu = 10;
        %         if isa(Xp{1},'ttensor')
        %             % Find roots of poly of degree 3
        %             temp = (G1 - tensor(G2([1:prod(R(1,:))]'),R(1,:)).*G3)./(1-G3.^2+eps);
        %             Xp{1}.core = temp;
        %         elseif isa(Xp{1},'ktensor')
        plambda = [1-double(G3)^2 double(G2(1)*double(G3) - G1) 0 -mu];
        lda3 = roots(plambda);
        lda3 = max((real(lda3)));
        %Xp{1}.lambda = double((G1 - G2(1:prod(R(1,:))).*G3)./(1-G3.^2+eps));
        Xp{1}.lambda = lda3;
        %         end
        
        if isa(Xp{2},'ttensor')
            temp = (reshape(G2([1:prod(R(1,:))]'),R(1,:)) - G1.*G3)./(1-G3.^2+eps);
            G2([1:prod(R(1,:))]') = temp(:);
            Xp{2}.core = G2;
        elseif isa(Xp{2},'ktensor')
            Xp{2}.lambda = double((G2 - G1.*G3)./(1-G3.^2+eps));
        end
        
        
        
        %         if isa(Xp{2},'ttensor')
        %             G2(1) = G2(1) - lda3*G3;
        %             Xp{2}.core = G2;
        %         elseif isa(Xp{2},'ktensor')
        %             Xp{2}.lambda = double(G2(1)-lda3*G3);
        %         end
    end


    function Xp = fastupdatecore_minblock1(Y,Xp)
        % Update core tensors of block terms
        % Call after rotate factors to be independent
        % update core
        %G = W\reshape(Y.data,I(1)*I(2),I(3));
        
        UtV = {sum(Xp{1}.U{1}.*Xp{2}.U{1}(:,1:R(1,1)))' ...
            sum(Xp{1}.U{2}.*Xp{2}.U{2}(:,1:R(1,2)))' ...
            sum(Xp{1}.U{3}.*Xp{2}.U{3}(:,1:R(1,3)))'};
        
        G1 = (ttm(Y,Xp{1}.U,'t'));
        G2 = (ttm(Y,Xp{2}.U,'t'));
        
        G3 = (full(ktensor(UtV)));
        
        mu = 1e-1;
        
        Xp{1}.lambda = double((G1 - G2(1:prod(R(1,:))).*G3)./(1+mu-G3.^2+eps));
        
        if isa(Xp{2},'ttensor')
            temp = ((1)*reshape(G2([1:prod(R(1,:))]'),R(1,:)) - G1.*G3)./(1-G3.^2+eps);
            G2([1:prod(R(1,:))]') = temp(:);
            Xp{2}.core = G2;
        elseif isa(Xp{2},'ktensor')
            Xp{2}.lambda = double(((1)*G2 - G1.*G3)./(1-G3.^2+eps));
        end
        
        %         if isa(Xp{2},'ttensor')
        %             temp = ((1+mu)*reshape(G2([1:prod(R(1,:))]'),R(1,:)) - G1.*G3)./(1-G3.^2+eps+mu);
        %             G2([1:prod(R(1,:))]') = temp(:);
        %             Xp{2}.core = G2;
        %         elseif isa(Xp{2},'ktensor')
        %             Xp{2}.lambda = double(((1+mu)*G2 - G1.*G3)./(1-G3.^2+eps+mu));
        %         end
        
        
        
        %         if isa(Xp{2},'ttensor')
        %             G2(1) = G2(1) - lda3*G3;
        %             Xp{2}.core = G2;
        %         elseif isa(Xp{2},'ktensor')
        %             Xp{2}.lambda = double(G2(1)-lda3*G3);
        %         end
    end

    function Xp = fastupdatecore_maxblock2(Y,Xp)
        % Update core tensors of block terms
        % Call after rotate factors to be independent
        % update core
        %G = W\reshape(Y.data,I(1)*I(2),I(3));
        
        UtV = {sum(Xp{1}.U{1}.*Xp{2}.U{1}(:,1:R(1,1)))' ...
            sum(Xp{1}.U{2}.*Xp{2}.U{2}(:,1:R(1,2)))' ...
            sum(Xp{1}.U{3}.*Xp{2}.U{3}(:,1:R(1,3)))'};
        
        G1 = (ttm(Y,Xp{1}.U,'t'));
        G2 = (ttm(Y,Xp{2}.U,'t'));
        
        G3 = (full(ktensor(UtV)));
        
        mu = 10;
        
        plambda = [1+mu-double(G3)^2 double(G2(1)*double(G3) - (1+mu)*G1) 0 -mu*(1+mu)];
        lda3 = roots(plambda);
        lda3 = max((real(lda3)));
        %Xp{1}.lambda = double((G1 - G2(1:prod(R(1,:))).*G3)./(1-G3.^2+eps));
        Xp{1}.lambda = lda3;
        %         end
        
        if isa(Xp{2},'ttensor')
            temp = (reshape(G2([1:prod(R(1,:))]'),R(1,:)) - G1.*G3)./(1-G3.^2+eps);
            G2([1:prod(R(1,:))]') = temp(:);
            Xp{2}.core = G2;
        elseif isa(Xp{2},'ktensor')
            Xp{2}.lambda = double((G2 - G1.*G3)./(1-G3.^2+eps));
        end
        
        
        
        %         if isa(Xp{2},'ttensor')
        %             G2(1) = (G2(1) - lda3*G3);
        %             Xp{2}.core = G2/(1+mu);
        %         elseif isa(Xp{2},'ktensor')
        %             Xp{2}.lambda = double(G2(1)-lda3*G3)/(1+mu);
        %         end
    end


    %% Fast update factors without computing core tensors
    function Xp = als_rank1R_v3(Y,Xp)
        % This ALS algorithm is used to update factor matrices when the
        % first block Xp{1} is of rank-1.
        %         Xp = rotatefactor(Xp);
        for n2 = 1:N
            %% CORRECT FORM %% Fast update when block 1 has rank of R1 = 1
            %lda = Xp{1}.lambda;
%             if isa(Xp{2},'ttensor')
%                 Gn = double(tenmat(Xp{2}.core,n2));
%             else
%                 Gn = Xp{2}.lambda;
%             end
            lambda = Xp{1}.lambda;
            rhon = prod(cellfun(@(x,y) x'*y(:,1),Xp{1}.U([1:n2-1 n2+1:N]),Xp{2}.U([1:n2-1 n2+1:N])));
            rho = rhon * Xp{1}.U{n2}'*Xp{2}.U{n2}(:,1);
            
            sn = double(ttv(Y,Xp{1}.U,[1:n2-1 n2+1:N]));
            Tn = double(tenmat(ttm(Y,Xp{2}.U,[1:n2-1 n2+1:N],'t'),n2));
          
            Gn = Xp{2}.U{n2}'*Tn;Gn(1) = Gn(1) - rho*lambda;
            Cg = Gn*Gn';
            Vn = (Tn*Gn' - rhon*sn*Gn(:,1)')/(Cg - rhon^2 * Gn(:,1)*Gn(:,1)');
            an = (sn - rhon*Vn*Gn(:,1))/Xp{1}.lambda;
            
%             wn = Gn(:,1);
%             term2 = bsxfun(@plus,wn,wn');
%             term2(1) = term2(1) - lambda*rho;
%             term2 = (1-rhon^2)*lambda*rho * term2;
%             
%             Num = Tn*Gn' - lambda*rho * 
%             Cg = Gn*Gn' - term2 - rho*wn*wn';
%             
%             Vn = /Cg;
%             an = (T1 - q*Vn*Gn(:,1))/Xp{1}.lambda;
            
            % Block 1
            ell1 = norm(an);
            Xp{1}.U{n2} = an/ell1;
            Xp{1}.lambda = Xp{1}.lambda * ell1;
            
            % decorrelate factors among block terms
            [Vn,rU] = qr(Vn,0);
            [v,rv] = qr(Vn'*Xp{1}.U{n2});
            
            Xp{2}.U{n2} = Vn * v;
            if isa(Xp{2},'ttensor')
                Xp{2}.core = ipermute(tensor(Gn,R(2,[n2 1:n2-1 n2+1:N])),[n2 1:n2-1 n2+1:N]);
                Xp{2}.core = ttm(Xp{2}.core,v'*rU,n2);
            elseif isa(Xp{2},'ktensor')
                Xp{2}.lambda = Xp{2}.lambda * v'*rU;
            end
            
        end
        Xp= fastupdatecore(Y,Xp);
    end


end


%*******************************************************************************
function [A,B,C] = bcdLrMrNr_lsearch(A,B,C,dA,dB,dC,D,X,lsearch,it,L_vec,M_vec,N_vec)
%BCDLMN_lsearch Line Search for BCD-(L,M,N)
%   New loading A, B, C are computed
%   from their previous values and from the search directions dA, dB, dC,
%   as follows:
%
%      A <- A + mu * dA
%      B <- B + mu * dB
%      C <- C + mu * dC
%
%   Line Search for bcdLMN can for instance be used in gradient-based
%   algorithms or with Alternating Least Squares (ALS), in order to speed
%   up convergence.
%   The core tensors Dr are kept constant, although they could in principle
%   also be interpolated.
%
%   For instance, if Line Search is used with ALS, the search directions
%   may be defined outside this function by dA=A1-A, dB=B1-B, dC=C1-C and
%   the input matrices by A=A1, B=B1 and C=C1 where (A1,B1,C1) denote
%   conditional least squares estimates at the current ALS iteration it,
%   whereas (A,B,C) denote old estimates, such that
%
%      A <- A + mu*(A1-A)
%      B <- B + mu*(B1-B)
%      C <- C + mu*(C1-C)
%
%   This means that, in the context of ALS, setting mu=1 annihilates the
%   line search, since A, B and C will be set to their current values A1,
%   B1 and C1.
%
%   For instance, if Line Search is used with gradient-based algorithms,
%   the search directions dA, dB, dC, could be (- gradient) of the cost
%   function wrt A, B and C, respectively, in a gradient descent
%   algorithm, or other more sophisticated search directions like in a
%   conjugate gradient algorithm.
%
%   Several choices of mu are possible:
%
%      - if lsearch = 'none' then no interpolation is done (in the sense of
%        lsearch for als), i.e. we enforce mu=1.
%      - if lsearch = 'lsh' (Line Search proposed by Harshman), then mu is
%        fixed to 1.25 and if the interpolated matrices do not decrease
%        the cost function, then mu=1 is enforced.
%      - if lsearch = 'lsb' (Line Search proposed by Bro), then mu is fixed
%        to it^(1/3) and if the interpolated matrices do not decrease
%        the cost function, then mu=1 is enforced.
%      - if lsearch = 'elsr' (Exact Line Search with real step), then we
%        seek for the optimal real-valued step mu that minimizes the cost
%        function. This amounts to minimizing a polynomial of degree 6 in
%        mu. It can also be used for complex-valued data.
%      - if lsearch = 'elsc' (Exact Line Search with complex step), then we
%        seek for the optimal complex-valued step mu that minimizes the
%        cost function. If the data are real-valued, the step has to be
%        real-valued, and in this case elsc is replaced by elsr.
%
%   INPUTS:
%
%      - A,B,C: estimates in cell format of the loading matrices A,B,C at it-1
%      - D is the cell that holds the set of R current core tensors Dr,
%      - dA,dB,dC: search directions in cell format
%      - X: JIxK matrix unfolding of the observed tensor X
%      - lsearch: = 'none', 'lsh','lsb', 'elsr' or 'elsc'
%      - it is the iteration step number
%      - L_vec: the way A is partitioned
%      - M_vec: the way B is partitioned
%      - N_vec: the way C is partitioned
%
%   OUTPUTS:
%      - Updated loadings A,B,C in cell format

%   Copyright 2010
%   Version: 09-07-10
%   Authors: Dimitri Nion (dimitri.nion@gmail.com),
%
%   References:
%   [1] R.A. Harshman, "Foundations of the PARAFAC procedure: Model and
%       Conditions for an explanatory Multi-mode Factor Analysis", UCLA
%       Working Papers in Phonetics, vol.16, pp 1-84, 1970.
%   [2] R. Bro, "Multi-Way Analysis in the Food Industry: Models,
%       Algorithms, and Applications", PhD. dissertation, University of
%       Amsterdam, 1998.
%   [3] M. Rajih, P. Comon and R.A. Harshman, "Enhanced Line Search: A
%       Novel Method to Accelerate  PARAFAC", SIAM Journal Matrix Anal. and
%       Appl. (SIMAX), Volume 30 , Issue 3, pp. 1128-1147, Sept 2008.
%   [4] D. Nion and L. De Lathauwer, "An Enhanced Line Search Scheme for
%       Complex-Valued Tensor Decompositions. Application in DS-CDMA",
%       Signal Processing, vol. 88, issue 3, pp. 749-755, March 2008.


it_start=3;   % Line Search will start after niter_start iterations (at least 2)
if isreal(X) && strcmp(lsearch,'elsc')==1; lsearch='elsr';end

% Convert inputs from cell format to matrix format
A=cell2mat(A);dA=cell2mat(dA);
B=cell2mat(B);dB=cell2mat(dB);
C=cell2mat(C);dC=cell2mat(dC);

R=length(L_vec);
RLM=sum(L_vec.*M_vec);
RN=sum(N_vec);
RLMc=[0,cumsum(L_vec.*M_vec)];
RNc=[0,cumsum(N_vec)];
Dm=zeros(RLM,RN);
for r=1:R
    Dm(RLMc(r)+1:RLMc(r+1),RNc(r)+1:RNc(r+1))=tens2mat(D{r},2);
end

% lsearch='none', i.e., standard ALS
if strcmp(lsearch,'none')==1
    mu=1;
    A=A+mu*dA;
    B=B+mu*dB;
    C=C+mu*dC;
    % lsearch='lsh'
elseif strcmp(lsearch,'lsh')==1
    if it<it_start
        mu=1;
        A=A+mu*dA;
        B=B+mu*dB;
        C=C+mu*dC;
    else
        % Compute phi with mu=1
        A=A+dA;
        B=B+dB;
        C=C+dC;
        phi=norm(X-kr_part(B,A,M_vec,L_vec)*Dm*C.','fro');
        % Compute phi with mu=1.25
        mu=1.25;
        Am=A+mu*dA;
        Bm=B+mu*dB;
        Cm=C+mu*dC;
        phim=norm(X-kr_part(Bm,Am,M_vec,L_vec)*Dm*Cm.','fro');
        % Accept or Reject Interpolation
        if phim < phi     % accept
            A=Am;
            B=Bm;
            C=Cm;
        end
    end
    % lsearch='lsb'
elseif strcmp(lsearch,'lsb')==1
    if it<it_start
        mu=1;
        A=A+mu*dA;
        B=B+mu*dB;
        C=C+mu*dC;
    else
        
        % Compute phi with mu=1
        A=A+dA;
        B=B+dB;
        C=C+dC;
        phi=norm(X-kr_part(B,A,M_vec,L_vec)*Dm*C.','fro');
        % Compute phi with mu=it^(1/3)
        mu=it^(1/3);
        Am=A+mu*dA;
        Bm=B+mu*dB;
        Cm=C+mu*dC;
        phim=norm(X-kr_part(Bm,Am,M_vec,L_vec)*Dm*Cm.','fro');
        % Accept or Reject Interpolation
        if phim < phi     % accept
            A=Am;
            B=Bm;
            C=Cm;
        end
    end
    % lsearch='elsr', compute optimal real-valued mu
elseif strcmp(lsearch,'elsr')==1
    if it<it_start
        mu=1;
    else
        KdBdA=kr_part(dB,dA,M_vec,L_vec);
        KBdA=kr_part(B,dA,M_vec,L_vec);
        KdBA=kr_part(dB,A,M_vec,L_vec);
        KBA=kr_part(B,A,M_vec,L_vec);
        DmdC=Dm*dC.';
        DmC=Dm*C.';
        Mat3=KdBdA*DmdC;
        Mat2=KdBdA*DmC + (KBdA+KdBA)*DmdC;
        Mat1=KBA*DmdC + (KBdA+KdBA)*DmC;
        Mat0=KBA*DmC-X;
        M=[Mat3(:) Mat2(:) Mat1(:) Mat0(:)];
        H_mat=real(M'*M);
        % Now we define the coefficients of the 6th order polynomial
        d6=H_mat(1,1);
        d5=2*H_mat(1,2);
        d4=2*H_mat(1,3)+H_mat(2,2);
        d3=2*(H_mat(1,4)+H_mat(2,3));
        d2=2*H_mat(2,4)+H_mat(3,3);
        d1=2*H_mat(3,4);
        d0=H_mat(4,4);
        pol=[d6 d5 d4 d3 d2 d1 d0];
        pol_der=[6*d6 5*d5 4*d4 3*d3 2*d2 d1];
        sqrts=roots(pol_der);
        sqrts=sqrts(imag(sqrts)==0); % real roots
        sqrts=[sqrts;1];
        % Choice of optimal mu
        extremum=polyval(pol,sqrts);
        mu=sqrts(find(extremum==min(extremum),1));
        mu=mu(1);
    end
    A=A+mu*dA;
    B=B+mu*dB;
    C=C+mu*dC;
    
    
elseif strcmp(lsearch,'elsc')==1
    % Alternate between updates of modulus and argument of mu
    if it<it_start
        mu=1;
    else
        Tolelsc=1e-4;
        Niterelsc=50;
        KdBdA=kr_part(dB,dA,M_vec,L_vec);
        KBdA=kr_part(B,dA,M_vec,L_vec);
        KdBA=kr_part(dB,A,M_vec,L_vec);
        KBA=kr_part(B,A,M_vec,L_vec);
        DmdC=Dm*dC.';
        DmC=Dm*C.';
        Mat3=KdBdA*DmdC;
        Mat2=KdBdA*DmC + (KBdA+KdBA)*DmdC;
        Mat1=KBA*DmdC + (KBdA+KdBA)*DmC;
        Mat0=KBA*DmC-X;
        M=[Mat3(:) Mat2(:) Mat1(:) Mat0(:)];
        H_mat=M'*M;
        M1=real(H_mat);
        M2=imag(H_mat);
        % Initialization
        mu=1;
        r=abs(mu);
        b=angle(mu);
        t=tan(b/2);
        u=[mu^3 mu^2 mu 1].';
        phi_new=abs(u'*H_mat*u);
        phi0=phi_new;   % initial value of the cost function (with mu=1)
        phi_diff=phi_new;
        it_in=0;
        % Alternate between updates of modulus and angle
        while (phi_diff > Tolelsc) && (it_in < Niterelsc)
            it_in=it_in+1;
            phi_old=phi_new;
            
            % Polynomial expression as a function of r
            h6=M1(1,1);
            h5=2*M1(1,2)*cos(b)+2*M2(1,2)*sin(b);
            h4=M1(2,2)+2*M1(1,3)*cos(2*b)+2*M2(1,3)*sin(2*b);
            h3=2*M1(1,4)*cos(3*b)+2*M1(2,3)*cos(b)+2*M2(1,4)*sin(3*b)+2*M2(2,3)*sin(b);
            h2=M1(3,3)+2*M1(2,4)*cos(2*b)+2*M2(2,4)*sin(2*b);
            h1=2*M1(3,4)*cos(b)+2*M2(3,4)*sin(b);
            h0=M1(4,4);
            pol=[h6 h5 h4 h3 h2 h1 h0];
            pol_der=[6*h6 5*h5 4*h4 3*h3 2*h2 h1];
            sqrts=roots(pol_der);
            sqrts=sqrts(imag(sqrts)==0);
            sqrts=[sqrts;1];
            extremum=polyval(pol,sqrts);
            r=sqrts(find(extremum==min(extremum),1));
            r=r(1);
            
            % Polynomial expression as a function of t=tan(b/2)
            a1=2*r^3*M1(1,4);
            a2=2*r^4*M1(1,3)+2*r^2*M1(2,4);
            a3=2*r^5*M1(1,2)+2*r^3*M1(2,3)+2*r*M1(3,4);
            a4=r^6*M1(1,1)+r^4*M1(2,2)+r^2*M1(3,3)+ M1(4,4);
            b1=2*r^3*M2(1,4);
            b2=2*r^4*M2(1,3)+2*r^2*M2(2,4);
            b3=2*r^5*M2(1,2)+2*r^3*M2(2,3)+2*r*M2(3,4);
            % Numerator coefficients
            d6=-a1+a2-a3+a4;
            d5=6*b1-4*b2+2*b3;
            d4=15*a1-5*a2-a3+3*a4;
            d3=-20*b1+4*b3;
            d2=-15*a1-5*a2+a3+3*a4;
            d1=6*b1+4*b2+2*b3;
            d0=a1+a2+a3+a4;
            % Denominator
            e6=-3*b1+2*b2-b3;
            e5=-18*a1+8*a2-2*a3;
            e4=45*b1-10*b2-b3;
            e3=60*a1-4*a3;
            e2=-45*b1-10*b2+b3;
            e1=-18*a1-8*a2-2*a3;
            e0=3*b1+2*b2+b3;
            Q=[e6 e5 e4 e3 e2 e1 e0];
            sqrts=roots(Q);
            sqrts=sqrts(imag(sqrts)==0);
            extremum=(1./(1+sqrts.^2).^3).*(d6*sqrts.^6+d5*sqrts.^5+d4*sqrts.^4+...
                d3*sqrts.^3+d2*sqrts.^2+d1*sqrts+d0*ones(length(sqrts),1));
            b=sqrts(find(extremum==min(extremum)));
            b=b(1);
            b=2*atan(b);
            
            % updated mu
            mu=r*exp(1i*b);
            t=tan(b/2);
            u=[mu^3 mu^2 mu 1].';
            phi_new=abs(u'*H_mat*u);
            phi_diff=abs(phi_new-phi_old);
        end
        if phi_new>phi0; mu=1;end
    end
    A=A+mu*dA;
    B=B+mu*dB;
    C=C+mu*dC;
end

% Convert to cell format
A=mat2cell(A,size(A,1),L_vec);
B=mat2cell(B,size(B,1),M_vec);
C=mat2cell(C,size(C,1),N_vec);
end

%*******************************************************************************
function [X_mat]=tens2mat(X,mode)
%TENS2MAT Matrix unfoldings of a 3rd order tensor X along a given mode
% INPUTS: - X : tensor of size (IxJxK)
%         - mode = 1 or 2 or 3
% OUTPUTS: X_mat is the matrix unfolding representation of X
% if mode==1:  X_mat is IKxJ  (i=1,...,I, is the slowly varying index)
% if mode==2:  X_mat is JIxK  (j=1,...,J, is the slowly varying index)
% if mode==3   X_mat is KJxI  (k=1,...,K  is the slowly varying index)
[I,J,K]=size(X);
if mode==1
    X_mat=reshape(permute(X,[3 1 2]),I*K,J);
elseif mode==2
    X_mat=reshape(X,J*I,K);
elseif mode==3
    X_mat=reshape(permute(X,[2 3 1]),K*J,I);
else
    error('Input argument mode must be 1, 2 or 3');
end
end

%*******************************************************************************
function [X]=mat2tens(X_mat,mode,size_vec)
%MAT2TENS Tensorization of a matrix (reciprocal of tens2mat)
if mode<1 || mode >3
    error ('Input argument mode must be a 1 , 2 or 3')
end
I=size_vec(1);
J=size_vec(2);
K=size_vec(3);
if mode==1
    X=permute(reshape(X_mat,K,I,J),[2 3 1]);
elseif mode==2
    X=reshape(X_mat,I,J,K);
elseif mode==3
    X=permute(reshape(X_mat,J,K,I),[3 1 2]);
end
end

%%
%*******************************************************************************
function [A,B,C,D] = bcdLrMrNr_lsearch2(A,B,C,dA,dB,dC,D,X,lsearch,it,L_vec,M_vec,N_vec,normX)
%BCDLMN_lsearch Line Search for BCD-(L,M,N)
%   New loading A, B, C are computed
%   from their previous values and from the search directions dA, dB, dC,
%   as follows:
%
%      A <- A + mu * dA
%      B <- B + mu * dB
%      C <- C + mu * dC
%
%   Line Search for bcdLMN can for instance be used in gradient-based
%   algorithms or with Alternating Least Squares (ALS), in order to speed
%   up convergence.
%   The core tensors Dr are kept constant, although they could in principle
%   also be interpolated.
%
%   For instance, if Line Search is used with ALS, the search directions
%   may be defined outside this function by dA=A1-A, dB=B1-B, dC=C1-C and
%   the input matrices by A=A1, B=B1 and C=C1 where (A1,B1,C1) denote
%   conditional least squares estimates at the current ALS iteration it,
%   whereas (A,B,C) denote old estimates, such that
%
%      A <- A + mu*(A1-A)
%      B <- B + mu*(B1-B)
%      C <- C + mu*(C1-C)
%
%   This means that, in the context of ALS, setting mu=1 annihilates the
%   line search, since A, B and C will be set to their current values A1,
%   B1 and C1.
%
%   For instance, if Line Search is used with gradient-based algorithms,
%   the search directions dA, dB, dC, could be (- gradient) of the cost
%   function wrt A, B and C, respectively, in a gradient descent
%   algorithm, or other more sophisticated search directions like in a
%   conjugate gradient algorithm.
%
%   Several choices of mu are possible:
%
%      - if lsearch = 'none' then no interpolation is done (in the sense of
%        lsearch for als), i.e. we enforce mu=1.
%      - if lsearch = 'lsh' (Line Search proposed by Harshman), then mu is
%        fixed to 1.25 and if the interpolated matrices do not decrease
%        the cost function, then mu=1 is enforced.
%      - if lsearch = 'lsb' (Line Search proposed by Bro), then mu is fixed
%        to it^(1/3) and if the interpolated matrices do not decrease
%        the cost function, then mu=1 is enforced.
%      - if lsearch = 'elsr' (Exact Line Search with real step), then we
%        seek for the optimal real-valued step mu that minimizes the cost
%        function. This amounts to minimizing a polynomial of degree 6 in
%        mu. It can also be used for complex-valued data.
%      - if lsearch = 'elsc' (Exact Line Search with complex step), then we
%        seek for the optimal complex-valued step mu that minimizes the
%        cost function. If the data are real-valued, the step has to be
%        real-valued, and in this case elsc is replaced by elsr.
%
%   INPUTS:
%
%      - A,B,C: estimates in cell format of the loading matrices A,B,C at it-1
%      - D is the cell that holds the set of R current core tensors Dr,
%      - dA,dB,dC: search directions in cell format
%      - X: JIxK matrix unfolding of the observed tensor X
%      - lsearch: = 'none', 'lsh','lsb', 'elsr' or 'elsc'
%      - it is the iteration step number
%      - L_vec: the way A is partitioned
%      - M_vec: the way B is partitioned
%      - N_vec: the way C is partitioned
%
%   OUTPUTS:
%      - Updated loadings A,B,C in cell format

%   Copyright 2010
%   Version: 09-07-10
%   Authors: Dimitri Nion (dimitri.nion@gmail.com),
%
%   References:
%   [1] R.A. Harshman, "Foundations of the PARAFAC procedure: Model and
%       Conditions for an explanatory Multi-mode Factor Analysis", UCLA
%       Working Papers in Phonetics, vol.16, pp 1-84, 1970.
%   [2] R. Bro, "Multi-Way Analysis in the Food Industry: Models,
%       Algorithms, and Applications", PhD. dissertation, University of
%       Amsterdam, 1998.
%   [3] M. Rajih, P. Comon and R.A. Harshman, "Enhanced Line Search: A
%       Novel Method to Accelerate  PARAFAC", SIAM Journal Matrix Anal. and
%       Appl. (SIMAX), Volume 30 , Issue 3, pp. 1128-1147, Sept 2008.
%   [4] D. Nion and L. De Lathauwer, "An Enhanced Line Search Scheme for
%       Complex-Valued Tensor Decompositions. Application in DS-CDMA",
%       Signal Processing, vol. 88, issue 3, pp. 749-755, March 2008.


it_start=3;   % Line Search will start after niter_start iterations (at least 2)
if isreal(X) && strcmp(lsearch,'elsc')==1; lsearch='elsr';end

% Convert inputs from cell format to matrix format
A=cell2mat(A);dA=cell2mat(dA);
B=cell2mat(B);dB=cell2mat(dB);
C=cell2mat(C);dC=cell2mat(dC);

R=length(L_vec);
RLM=sum(L_vec.*M_vec);
RN=sum(N_vec);
RLMc=[0,cumsum(L_vec.*M_vec)];
RNc=[0,cumsum(N_vec)];
Dm=zeros(RLM,RN);
for r=1:R
    Dm(RLMc(r)+1:RLMc(r+1),RNc(r)+1:RNc(r+1))=tens2mat(D{r},2);
end

% lsearch='none', i.e., standard ALS
if strcmp(lsearch,'none')==1
    mu=1;
    A=A+mu*dA;
    B=B+mu*dB;
    C=C+mu*dC;
    % lsearch='lsh'
elseif strcmp(lsearch,'lsh')==1
    if it<it_start
        mu=1;
        A=A+mu*dA;
        B=B+mu*dB;
        C=C+mu*dC;
    else
        % Compute phi with mu=1
        A=A+dA;
        B=B+dB;
        C=C+dC;
        
        phi = Frobnorm(mat2cell(A,size(A,1),L_vec),...
            mat2cell(B,size(B,1),M_vec),...
            mat2cell(C,size(C,1),N_vec),D,normX);
        %phi=norm(X-kr_part(B,A,M_vec,L_vec)*Dm*C.','fro');
        % Compute phi with mu=1.25
        mu=1.25;
        Am=A+mu*dA;
        Bm=B+mu*dB;
        Cm=C+mu*dC;
        phim = Frobnorm(mat2cell(Am,size(A,1),L_vec),...
            mat2cell(Bm,size(B,1),M_vec),...
            mat2cell(Cm,size(C,1),N_vec),D,normX);
        %phim=norm(X-kr_part(Bm,Am,M_vec,L_vec)*Dm*Cm.','fro');
        % Accept or Reject Interpolation
        if phim < phi     % accept
            A=Am;
            B=Bm;
            C=Cm;
        end
    end
    % lsearch='lsb'
elseif strcmp(lsearch,'lsb')==1
    if it<it_start
        mu=1;
        A=A+mu*dA;
        B=B+mu*dB;
        C=C+mu*dC;
    else
        
        % Compute phi with mu=1
        A=A+dA;
        B=B+dB;
        C=C+dC;
        phi = Frobnorm(mat2cell(A,size(A,1),L_vec),...
            mat2cell(B,size(B,1),M_vec),...
            mat2cell(C,size(C,1),N_vec),D,normX);
        %phi=norm(X-kr_part(B,A,M_vec,L_vec)*Dm*C.','fro');
        % Compute phi with mu=it^(1/3)
        mu=it^(1/3);
        Am=A+mu*dA;
        Bm=B+mu*dB;
        Cm=C+mu*dC;
        
        phim = Frobnorm(mat2cell(Am,size(A,1),L_vec),...
            mat2cell(Bm,size(B,1),M_vec),...
            mat2cell(Cm,size(C,1),N_vec),D,normX);
        %phim=norm(X-kr_part(Bm,Am,M_vec,L_vec)*Dm*Cm.','fro');
        % Accept or Reject Interpolation
        if phim < phi     % accept
            A=Am;
            B=Bm;
            C=Cm;
        end
    end
    % lsearch='elsr', compute optimal real-valued mu
elseif strcmp(lsearch,'elsr')==1
    if it<it_start
        mu=1;
    else
        KdBdA=kr_part(dB,dA,M_vec,L_vec);
        KBdA=kr_part(B,dA,M_vec,L_vec);
        KdBA=kr_part(dB,A,M_vec,L_vec);
        KBA=kr_part(B,A,M_vec,L_vec);
        DmdC=Dm*dC.';
        DmC=Dm*C.';
        Mat3=KdBdA*DmdC;
        Mat2=KdBdA*DmC + (KBdA+KdBA)*DmdC;
        Mat1=KBA*DmdC + (KBdA+KdBA)*DmC;
        Mat0=KBA*DmC-X;
        M=[Mat3(:) Mat2(:) Mat1(:) Mat0(:)];
        H_mat=real(M'*M);
        % Now we define the coefficients of the 6th order polynomial
        d6=H_mat(1,1);
        d5=2*H_mat(1,2);
        d4=2*H_mat(1,3)+H_mat(2,2);
        d3=2*(H_mat(1,4)+H_mat(2,3));
        d2=2*H_mat(2,4)+H_mat(3,3);
        d1=2*H_mat(3,4);
        d0=H_mat(4,4);
        pol=[d6 d5 d4 d3 d2 d1 d0];
        pol_der=[6*d6 5*d5 4*d4 3*d3 2*d2 d1];
        sqrts=roots(pol_der);
        sqrts=sqrts(imag(sqrts)==0); % real roots
        sqrts=[sqrts;1];
        % Choice of optimal mu
        extremum=polyval(pol,sqrts);
        mu=sqrts(find(extremum==min(extremum),1));
        mu=mu(1);
    end
    A=A+mu*dA;
    B=B+mu*dB;
    C=C+mu*dC;
    
    
elseif strcmp(lsearch,'elsc')==1
    % Alternate between updates of modulus and argument of mu
    if it<it_start
        mu=1;
    else
        Tolelsc=1e-4;
        Niterelsc=50;
        KdBdA=kr_part(dB,dA,M_vec,L_vec);
        KBdA=kr_part(B,dA,M_vec,L_vec);
        KdBA=kr_part(dB,A,M_vec,L_vec);
        KBA=kr_part(B,A,M_vec,L_vec);
        DmdC=Dm*dC.';
        DmC=Dm*C.';
        Mat3=KdBdA*DmdC;
        Mat2=KdBdA*DmC + (KBdA+KdBA)*DmdC;
        Mat1=KBA*DmdC + (KBdA+KdBA)*DmC;
        Mat0=KBA*DmC-X;
        M=[Mat3(:) Mat2(:) Mat1(:) Mat0(:)];
        H_mat=M'*M;
        M1=real(H_mat);
        M2=imag(H_mat);
        % Initialization
        mu=1;
        r=abs(mu);
        b=angle(mu);
        t=tan(b/2);
        u=[mu^3 mu^2 mu 1].';
        phi_new=abs(u'*H_mat*u);
        phi0=phi_new;   % initial value of the cost function (with mu=1)
        phi_diff=phi_new;
        it_in=0;
        % Alternate between updates of modulus and angle
        while (phi_diff > Tolelsc) && (it_in < Niterelsc)
            it_in=it_in+1;
            phi_old=phi_new;
            
            % Polynomial expression as a function of r
            h6=M1(1,1);
            h5=2*M1(1,2)*cos(b)+2*M2(1,2)*sin(b);
            h4=M1(2,2)+2*M1(1,3)*cos(2*b)+2*M2(1,3)*sin(2*b);
            h3=2*M1(1,4)*cos(3*b)+2*M1(2,3)*cos(b)+2*M2(1,4)*sin(3*b)+2*M2(2,3)*sin(b);
            h2=M1(3,3)+2*M1(2,4)*cos(2*b)+2*M2(2,4)*sin(2*b);
            h1=2*M1(3,4)*cos(b)+2*M2(3,4)*sin(b);
            h0=M1(4,4);
            pol=[h6 h5 h4 h3 h2 h1 h0];
            pol_der=[6*h6 5*h5 4*h4 3*h3 2*h2 h1];
            sqrts=roots(pol_der);
            sqrts=sqrts(imag(sqrts)==0);
            sqrts=[sqrts;1];
            extremum=polyval(pol,sqrts);
            r=sqrts(find(extremum==min(extremum),1));
            r=r(1);
            
            % Polynomial expression as a function of t=tan(b/2)
            a1=2*r^3*M1(1,4);
            a2=2*r^4*M1(1,3)+2*r^2*M1(2,4);
            a3=2*r^5*M1(1,2)+2*r^3*M1(2,3)+2*r*M1(3,4);
            a4=r^6*M1(1,1)+r^4*M1(2,2)+r^2*M1(3,3)+ M1(4,4);
            b1=2*r^3*M2(1,4);
            b2=2*r^4*M2(1,3)+2*r^2*M2(2,4);
            b3=2*r^5*M2(1,2)+2*r^3*M2(2,3)+2*r*M2(3,4);
            % Numerator coefficients
            d6=-a1+a2-a3+a4;
            d5=6*b1-4*b2+2*b3;
            d4=15*a1-5*a2-a3+3*a4;
            d3=-20*b1+4*b3;
            d2=-15*a1-5*a2+a3+3*a4;
            d1=6*b1+4*b2+2*b3;
            d0=a1+a2+a3+a4;
            % Denominator
            e6=-3*b1+2*b2-b3;
            e5=-18*a1+8*a2-2*a3;
            e4=45*b1-10*b2-b3;
            e3=60*a1-4*a3;
            e2=-45*b1-10*b2+b3;
            e1=-18*a1-8*a2-2*a3;
            e0=3*b1+2*b2+b3;
            Q=[e6 e5 e4 e3 e2 e1 e0];
            sqrts=roots(Q);
            sqrts=sqrts(imag(sqrts)==0);
            extremum=(1./(1+sqrts.^2).^3).*(d6*sqrts.^6+d5*sqrts.^5+d4*sqrts.^4+...
                d3*sqrts.^3+d2*sqrts.^2+d1*sqrts+d0*ones(length(sqrts),1));
            b=sqrts(find(extremum==min(extremum)));
            b=b(1);
            b=2*atan(b);
            
            % updated mu
            mu=r*exp(1i*b);
            t=tan(b/2);
            u=[mu^3 mu^2 mu 1].';
            phi_new=abs(u'*H_mat*u);
            phi_diff=abs(phi_new-phi_old);
        end
        if phi_new>phi0; mu=1;end
    end
    A=A+mu*dA;
    B=B+mu*dB;
    C=C+mu*dC;
end

% Convert to cell format
A=mat2cell(A,size(A,1),L_vec);
B=mat2cell(B,size(B,1),M_vec);
C=mat2cell(C,size(C,1),N_vec);

[A,B,C,D] = rotatefactor(A,B,C,D);
end


%***********************************************************************
function error = Frobnorm(A_cell,B_cell,C_cell,D_cell,normX)
R(1,1) = size(A_cell{1},2);
R(1,2) = size(B_cell{1},2);
R(1,3) = size(C_cell{1},2);

% check orthogonality
UtV = {A_cell{1}'*A_cell{2} ...
    B_cell{1}' * B_cell{2} ...
    C_cell{1}' * C_cell{2}};
orthfg = norm(tril(UtV{1},1)) + ...
    norm(tril(UtV{2},1)) + ...
    norm(tril(UtV{3},1));

if orthfg > 1e-6
    % Rotate factor matrices
    [A_cell,B_cell,C_cell,D_cell,UtV] = rotatefactor(A_cell,B_cell,C_cell,D_cell);
else
%     UtV = cellfun(@(x) diag(x),UtV,'uni',0);
end

% UtV = {sum(A_cell{1}.*A_cell{2}(:,1:R(1,1)))' ...
%     sum(B_cell{1}.*B_cell{2}(:,1:R(1,2)))' ...
%     sum(C_cell{1}.*C_cell{2}(:,1:R(1,3)))'};

H = double(full(ktensor(UtV)));
sZ = size(D_cell{1});
if numel(sZ) == 2
    sZ(3) = 1;
end
error = normX^2 - norm(D_cell{1}(:))^2 - norm(D_cell{2}(:))^2 - ...
    2 * D_cell{1}(:)' * reshape(D_cell{2}(1:sZ(1),1:sZ(2),1:sZ(3)).*H,[],1);
error = real(sqrt(error));
% error = sqrt(error)/normX;
end



% % *************************************************************************
% function [A_cell,B_cell,C_cell,D_cell,xC] = rotatefactor(A_cell,B_cell,C_cell,D_cell)
% if nargin == 3 
%     D_cell = [];
% end
% % Rotate factors
% for p2 = 1:2
%     % orthogonalize A
%     [A_cell{p2},rr] = qr(A_cell{p2},0);
%     if ~isempty(D_cell)
%         D_cell{p2} = tmprod(D_cell{p2},rr,1);
%     end
%     % orthogonalize B
%     [B_cell{p2},rr] = qr(B_cell{p2},0);
%     if ~isempty(D_cell)
%         D_cell{p2} = tmprod(D_cell{p2},rr,2);
%     end
%     
%     % orthogonalize C
%     [C_cell{p2},rr] = qr(C_cell{p2},0);
%     if ~isempty(D_cell)
%         D_cell{p2} = tmprod(D_cell{p2},rr,3);
%     end
% end
% 
% % Rotate factors to be orthogonal
% 
% [u,xC{1},v] = svd(A_cell{1}'*A_cell{2},0);    
% A_cell{1} = A_cell{1} * u;
% A_cell{2} = A_cell{2} * v;
% 
% if ~isempty(D_cell)
%     D_cell{1} = tmprod(D_cell{1},u',1);
%     D_cell{2} = tmprod(D_cell{2},v',1);
% end
% 
% [u,xC{2},v] = svd(B_cell{1}'*B_cell{2},0);    
% B_cell{1} = B_cell{1} * u;
% B_cell{2} = B_cell{2} * v;
%     
% if ~isempty(D_cell)
%     D_cell{1} = tmprod(D_cell{1},u',2);
%     D_cell{2} = tmprod(D_cell{2},v',2);
% end
% 
% [u,xC{3},v] = svd(C_cell{1}'*C_cell{2},0);    
% C_cell{1} = C_cell{1} * u;
% C_cell{2} = C_cell{2} * v;
%     
% if ~isempty(D_cell)
%     D_cell{1} = tmprod(D_cell{1},u',3);
%     D_cell{2} = tmprod(D_cell{2},v',3);
% end
% % xC = cellfun(@(x) diag(x),xC,'uni',0);
% end
