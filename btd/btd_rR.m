function [Xp,Err] = btd_rR(Y,R,opts)
% TENSORBOX, 2018

%% Set algorithm parameters from input or by using defaults
if ~exist('opts','var'), opts = struct; end


param = inputParser;
param.KeepUnmatched = true;

param.addOptional('init','block',@(x) (iscell(x) ||...
    ismember(x,{'ajd' 'block' 'td'})));
param.addOptional('updateterm',true);
param.addOptional('maxiters',1000);
param.addOptional('updatefactor',false);
param.addOptional('updateAG','none',@(x) ismember(x,{'none' 'als' 'svd'}));
param.addOptional('rotatefactor',false);
param.addOptional('correctfactor',true);
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
P = size(R,1);

%% Initialization
Xp = bcd_init;

normY = norm(Y);
abserror = Frobnorm(Xp,normY);
err = abserror/normY;
errold = err;
Err = err;
 
sta = [0 ;prod(R,2)];

%%
done = false; cnt1 = 0;

while ~done && ( cnt1 < param.maxiters)
    cnt1 = cnt1+1;
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
        Xp= alsupdate(Y,Xp);
        %Xp = fastals(Y,Xp);
%         Xp = fastals_blk(Y,Xp);
        % Xp = als_procrustes(Y,Xp);
    end
    
    % Alternating rules update block terms
    if param.updateterm
        Yr = Y - Yh;
        for p = 1:P
            Yr = Yr + tensor(Xp{p});
            Xp{p} = mtucker_als(Yr,R(p,:),opt_hooi,Xp{p}.U);
            Yr = Yr - tensor(Xp{p});
        end
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
%     Yh = 0;
%     for p3 = 1:P
%         Yh = Yh + full(Xp{p3});
%     end
%     abserror = norm(Y-Yh);
    abserror = Frobnorm(Xp,normY);
    err = abserror/normY;
    
    Err = [Err err];
    fprintf('Err %d\n',err)
    if ~isinf(errold)
        done = (abs(err-errold) < errold * param.tol) || (abs(err-errold) < param.tol);
    end
    errold = err;
    
end

% Rotate factors
Xp = rotatefactor(Xp);

if P == 2
    Xp = fastupdatecore(Y,Xp);
else
    Xp = updatecore(Y,Xp);
end
 

%% *********************************************************************
    function abserror = Frobnorm(Xp,normY)
        % Fast Computation of the Frobenius norm of the residual tensor Y -
        % Yhat
        % normY = norm(Y) is the Frobenius norm of Y.
        % Using with least squares algorithm, e.g., ALS
        
        abserror = 0;
        for p = 1:P
            abserror = abserror + norm(Xp{p})^2;
            for q = p+1:P
                abserror = abserror + 2*innerprod(full(ttm(Xp{p},Xp{q}.u,'t')),Xp{q}.core);
            end
        end
        abserror = normY^2 - abserror;
        abserror = sqrt(abserror);
    end

%% *********************************************************************

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
                
                [A0 B0 C0 S]=btd3(double(Ycpr.core),R(:,1),1);
                %[A0 B0 C0 S iter]=tedia2b(double(Ycpr.core),20);
            else
                %[A0 B0 C0 S iter]=tedia2b(double(Y),20);
                [A0 B0 C0 S]=btd3(double(Y),R(:,1),1);
            end
            
            A0 = inv(A0);
            B0 = inv(B0);
            C0 = inv(C0);
            
            % Sort components of A and B
            SS=sum(abs(S),3);
            [u,s,v] = svds(SS - mean(SS(:)),1);
            [u,ord1] = sort(u,'descend');
            [v,ord2] = sort(v,'descend');
            S = S(ord1,ord2,:);
            A0 = A0(:,ord1);
            B0 = B0(:,ord2);
            
            SS = squeeze(sum(abs(S),2));
            [u,s,v] = svds(SS - mean(SS(:)),1);
            %[u,ord1] = sort(u,'descend');
            [v,ord2] = sort(v,'descend');
            S = S(:,:,ord2);
            C0 = C0(:,ord2);
            
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
            
            
        elseif iscell(param.init)
            Xp = param.init;
        end
        
        % % Rotate factors
        Xp = rotatefactor(Xp);
        
        if P == 2
            Xp = fastupdatecore(Y,Xp);
        else
            Xp = updatecore(Y,Xp);
        end
        
    end

%% *********************************************************************

    function Xp = rotatefactor(Xp)
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
                Xp{1}.core = ttm(Xp{1}.core,u',n);
            elseif isa(Xp{1},'ktensor')
                Xp{1}.lambda = Xp{1}.lambda * u';
            end
            
            
            if isa(Xp{2},'ttensor')
                Xp{2}.core = ttm(Xp{2}.core,v',n);
            elseif isa(Xp{2},'ktensor')
                Xp{2}.lambda = Xp{2}.lambda * v';
            end
            
%             Xp{1}.core = ttm(Xp{1}.core,u',n2);
%             Xp{2}.core = ttm(Xp{2}.core,v',n2);
        end
        
    end

%% *********************************************************************
    
    function Xp= corr_deg(Xp)
        % Call after rotating factors
        for n2 = 1:N
            cdeg(n2,:) = diag(Xp{1}.U{n2}'*Xp{2}.U{n2});
            
            ix = find(cdeg(n2,:) > 0.99);
            if ~isempty(ix)
                cdeg(n2,ix) = cdeg(n2,ix)*.3;
%                 [u,s,v] = svd(Xp{1}.U{n2}'*Xp{2}.U{n2},0);
                [Unbar,rr] = qr(Xp{2}.U{n2});
                
                % Find Q
                Z = ttm(Xp{1}.core,Xp{1}.U,-n2);                
                Z = double(tenmat(Z,n2));
                temp = Unbar(:,size(Xp{2}.U{n2},2)+1:end)'*...
                    double(tenmat(Y,n2))*Z'*diag(sqrt(1-cdeg(n2,:).^2));
                [ut,st,vt] = svd(temp);
                Qn = ut*vt' * diag(sqrt(1-cdeg(n2,:).^2));
                
                Xp{1}.U{n2} = Xp{2}.U{n2} * diag(cdeg(n2,:)) + Unbar(:,size(Xp{2}.U{n2},2)+1:end)* Qn;
            end
            
        end
        
%         [ix,ix2] = find(cdeg > 0.99);
%         for ki = 1:numel(ix)
%             if ix(ki) == 1
%                 [A B1 S]=btd_ajd(Y.data,R(:,1));
%             end
%             if ix(ki) == 2
%                 [B A1 S]=btd_ajd(permute(Y.data,[2 1 3]),R(:,1));
%             end
%             if ix(ki) == 3
%                 [C A1 S]=btd_ajd(permute(Y.data,[3 1 2]),R(:,1));
%             end
%         end       
    end

%% *********************************************************************

    function Xp = updatecore(Y,Xp)
        % Update core tensors
        % update core
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
%         U = W'*W;
%         UtV = cellfun(@(x,y) x'*y,Xp{1}.U,Xp{2}.u,'uni',0);
        
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

%% *********************************************************************

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
            temp = (G1 - tensor(G2([1:prod(R(1,:))]'),R(1,:)).*G3)./(1-G3.^2+eps);
            Xp{1} = ttensor(temp,U);
        else 
            lambda = double((G1 - G2(1:prod(R(1,:))).*G3)./(1-G3.^2+eps));
            Xp{1} = ktensor(lambda,U);
        end
        
        if size(V{1},2) > 1 
            temp = (reshape(G2([1:prod(R(1,:))]'),R(1,:)) - G1.*G3)./(1-G3.^2+eps);
            G2([1:prod(R(1,:))]') = temp(:);
            Xp{2} = ttensor(G2,V);
        else 
            lambda = double((G2 - G1.*G3)./(1-G3.^2+eps));
            Xp{2} = ktensor(lambda,V);
        end
    end

    function Xp = fastupdatecore(Y,Xp)
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
        
        if isa(Xp{1},'ttensor')
            temp = (G1 - G2(1:R(1,1),1:R(1,2),1:R(1,3)).*G3)./(1-G3.^2+eps);
            Xp{1}.core = temp;
        elseif isa(Xp{1},'ktensor')
            Xp{1}.lambda = double((G1 - G2(1:R(1,1),1:R(1,2),1:R(1,3)).*G3)./(1-G3.^2+eps));
        end
        
        if isa(Xp{2},'ttensor')
            G2(1:R(1,1),1:R(1,2),1:R(1,3)) = (G2(1:R(1,1),1:R(1,2),1:R(1,3)) - G1.*G3)./(1-G3.^2+eps);
            Xp{2}.core = G2;
        elseif isa(Xp{2},'ktensor')
            Xp{2}.lambda = double((G2 - G1.*G3)./(1-G3.^2+eps));
        end
    end

%% *********************************************************************

    function Xp= alsupdate(Y,Xp)
        %--- Algorithm parameters
        comp='off';          % ='on' or ='off' to perform or not dimensionality reduction
        Tol1=1e-6;          % Tolerance
        MaxIt1=5;        % Max number of iterations
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
        %[A_est,B_est,C_est,D_est,phi,it1,it2,phi_als]=bcdLrMrNr_alsls_twobcks(Y.data,...
        %    R(:,1)',R(:,2)',R(:,3)',ls,comp,Tol1,MaxIt1,Tol2,MaxIt2,Ninit,A_init,B_init,C_init,D_init);
        for p2 = 1:P
            if numel(D_init{p2}) == 1
                Xp{p2} = ktensor(D_est{p2},{A_est{p2}, B_est{p2}, C_est{p2}});
            else
                Xp{p2} = ttensor(tensor(D_est{p2}),{A_est{p2}, B_est{p2}, C_est{p2}});
            end
        end
        
    end



   %% *********************************************************************
    function Xp = fastals_splithalf(Y,Xp)
        
        for p = 1:P
            % Estimate factors of the subtensor Xp while fixing other
            % tensors Xq, q # p
            
            % Construct the complementary tensor Xp_c
            for q = [1:p-1 p+1:P]
                
            end
        end
        
    end

%% *********************************************************************
    function Xp = fastals(Y,Xp)
        utu = cellfun(@(x,y) x'*y,Xp{1}.u,Xp{2}.u,'uni',0);

        for n2 = 1:N
            
%             if isa(Xp{1},'ttensor')
%                 Zp{1} = ttm(Xp{1}.core,Xp{1}.U,-n2);
%                 Zp{1} = double(tenmat(Zp{1},n2));
%             elseif isa(Xp{1},'ktensor')
%                 tt = khatrirao(Xp{1}.U([1:n2-1 n2+1:end]),'r')';
%                 Zp{1} = diag(Xp{1}.lambda) * tt;
%             end
%             if isa(Xp{2},'ttensor')
%                 Zp{2} = ttm(Xp{2}.core,Xp{2}.U,-n2);
%                 Zp{2} = double(tenmat(Zp{2},n2));
%             elseif isa(Xp{2},'ktensor')
%                 Zp{2} = diag(Xp{2}.lambda) * khatrirao(Xp{2}.U([1:n2-1 n2+1:end]),'r')';
%             end
%             
%             Un2 = double(tenmat(Y,n2))/[Zp{1}; Zp{2}];

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

    function Xp = als_procrustes(Y,Xp)
%         Xp = rotatefactor(Xp);
        for n2 = 1:N
%             if isa(Xp{1},'ttensor')
%                 Zp{1} = ttm(Xp{1}.core,Xp{1}.U,-n2);
%                 Zp{1} = double(tenmat(Zp{1},n2));
%             elseif isa(Xp{1},'ktensor')
%                 tt = khatrirao(Xp{1}.U([1:n2-1 n2+1:end]),'r')';
%                 Zp{1} = diag(Xp{1}.lambda) * tt;
%             end
%             
%             
%             if isa(Xp{2},'ttensor')
%                 Zp{2} = ttm(Xp{2}.core,Xp{2}.U,-n2);
%                 Zp{2} = double(tenmat(Zp{2},n2));
%             elseif isa(Xp{2},'ktensor')
%                 Zp{2} = diag(Xp{2}.lambda) * khatrirao(Xp{2}.U([1:n2-1 n2+1:end]),'r')';
%             end
%             
%             
%             Un = double(tenmat(Y,n2))/[Zp{1}; Zp{2}];
%             Un = mat2cell(Un,I(n2),R(:,n2)');
            
            % Fast update Un
            utu = cellfun(@(x,y) x'*y,Xp{1}.u,Xp{2}.u,'uni',0);
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
            
            
            
            %             for k2 = 1:5
            %
            %                 % Find s2
            %                 [U2bar,rr] = qr(Xp{2}.U{n});
            %                 U2bar(:,1:R(2,n)) = Xp{2}.U{n};
            %
            %                 YtAn = tenmat(ttm(Y,U2bar',n),n);
            %                 temp2 = YtAn(1:R(2,n),:) - Zp{2};
            %                 s = sum(temp2 .* Zp{1},2)./sum(Zp{1}.* Zp{1},2);
            %                 s = min(max(-1,s),1);
            %
            %                 % Find Q2
            %                 temp = YtAn(R(2,n)+1:end,:)*Zp{1}'*diag(sqrt(1-s.^2));
            %                 [ut,st,vt] = svd(temp);
            %                 Qn = ut*vt';
            %
            %                 Xp{1}.U{n} = U2bar * [diag(s); Qn * diag(sqrt(1-s.^2))];
            %
            %
            %                 % Find s1
            %                 [U1bar,rr] = qr(Xp{1}.U{n});
            %                 U1bar(:,1:R(1,n)) = Xp{1}.U{n};
            %
            %                 YtAn = tenmat(ttm(Y,U1bar',n),n);
            %                 temp2 = YtAn(1:R(1,n),:) - Zp{1};
            %                 s = sum(temp2 .* Zp{2},2)./sum(Zp{2}.* Zp{2},2);
            %                 s = min(max(-1,s),1);
            %
            %                 % Find Q1
            %                 temp = YtAn(R(1,n)+1:end,:)*Zp{2}'*diag(sqrt(1-s.^2));
            %                 [ut,st,vt] = svd(temp);
            %                 Qn = ut*vt';
            %
            %                 Xp{2}.U{n} = U1bar * [diag(s); Qn * diag(sqrt(1-s.^2))];
            %             end
            
        end
        
        if P == 2
            Xp = fastupdatecore(Y,Xp);
        else
            Xp = updatecore(Y,Xp);
        end
        
    end
end