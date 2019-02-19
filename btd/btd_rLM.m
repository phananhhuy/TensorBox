function [Xp,Err] = btd_rLM(Y,R,opts)
% TENSORBOX, 2018

%% Set algorithm parameters from input or by using defaults
if ~exist('opts','var'), opts = struct; end


param = inputParser;
param.KeepUnmatched = true;

param.addOptional('init','block',@(x) (iscell(x) ||...
    ismember(x,{'ajd' 'block'})));
param.addOptional('updateterm',true);
param.addOptional('maxiters',1000);
param.addOptional('updatefactor',false);
param.addOptional('rotatefactor',false);
param.addOptional('correctfactor',true);
param.addOptional('correcttwofactors',false);
param.addOptional('alsupdate',false);

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

%% Init
opt_hooi = struct('printitn',0,'init','nvecs','dimorder',1:N-1);

if strcmp(param.init,'block')    
    Xp = cell(P,1);
    Xp{1} = mtucker_als(Y,R(1,:),opt_hooi);
    for p = 2:P
        Yr = Y - tensor(Xp{1});
        Xp{p} = mtucker_als(Yr,R(p,:),opt_hooi);
        % Xp{p}.core =Xp{p}.core;
    end
    
elseif strcmp(param.init,'ajd')
    % init using joint diagonal block term
    [A B S]=btd(Y.data,R(:,1));
    A = inv(A);
    B = inv(B);
    A = mat2cell(A,size(A,1),R(1,:));
    B = mat2cell(B,size(B,1),R(2,:));
    
    Xp{1} = ttensor(tensor(S(1:R(1,1),1:R(1,2),:)),{A{1},B{1} eye(I(end))});
    Xp{2} = ttensor(tensor(S(R(1,1)+1:end,R(1,2)+1:end,:)),{A{2},B{2} eye(I(end))});
end
% Rotate factors
Xp = rotatefactor(Xp);

%G = updatecore(Y,Xp);
Xp= fastupdatecore(Y,Xp);

% Xp = Tp;
Yh = 0;
for p = 1:P
    Yh = Yh + tensor(Xp{p});
end

normY = norm(Y);
err = norm(Y - Yh)/normY;
%
errold = err;
Err = [err];

sta = [0 ;prod(R,2)];

%%
done = false; cnt1 = 0;
while ~done && ( cnt1 < param.maxiters)
    cnt1 = cnt1+1;
    
    if param.updatefactor
        Xp= alsupdate(Y,Xp);
    end
    
    if param.updateterm
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
    
    %% rotate An such that A1n^T * A2n = diag(s)
    if param.correctfactor
        Xp = rotatefactor(Xp);
        Xp = fastupdatecore(Y,Xp);
        
        %
        %         for n = 1:N-1
        %             [u,s,v] = svd(Xp{1}.U{n}'*Xp{2}.U{n},0);
        %             Xp{1}.U{n} = Xp{1}.U{n} * u;
        %             Xp{2}.U{n} = Xp{2}.U{n} * v;
        %
        %             Xp{1}.core = ttm(Xp{1}.core,u',n);
        %             Xp{2}.core = ttm(Xp{2}.core,v',n);
        %         end
        
        done2 = false;errold2 = errold;cnt = 0;
        while  ~done2 && ( cnt < 20)
            cnt = cnt + 1;
            
            %% Update block terms
            %             Yh = 0;
            %             for p = 1:P
            %                 Yh = Yh + tensor(Xp{p});
            %             end
            %             Yr = Y - Yh;
            %             for p = 1:2
            %                 Yr = Yr + tensor(Xp{p});
            %                 Xp{p} = mtucker_als(Yr,R(p,:),opt_hooi,Xp{p}.U);
            %                 Yr = Yr - tensor(Xp{p});
            %             end
            %             % update core
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
            %             G = mat2cell(G,prod(R,2),I(N));
            %             for p = 1:P
            %                 Xp{p}.core = tensor(reshape(G{p},[R(p,:) I(N)]));
            %             end
            
            
            %% Correct A1
            % Fix A2 and Update A1 = A2 diag(s)^2 + A2c *Q *diag(sqrt(1-s^2)
            % Find s and Q where [A2 A2c] are column space of Y2
          
            for n = 1:N-1
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
            Xp = fastupdatecore(Y,Xp);
            
            
            % Correct A2
            % Fix A1 and Update A2 = A1 diag(s)^2 + A1c *Q *diag(sqrt(1-s^2)
            % Find s and Q where [A1 A1c] are column space of Y1
           
            for n = 1:N-1
                
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
            Xp = fastupdatecore(Y,Xp);
            
            % Approximate error
            Yh = 0;
            for p = 1:P
                Yh = Yh + tensor(Xp{p});
            end
            err = norm(Y - Yh)/normY;
            %Err = [Err err];
            %fprintf('Err2 %d\n',err)
            if ~isinf(errold2)
                done2 = abs(err-errold2) < errold2 * 1e-4;
            end
            errold2 = err;
            
        end
    end
    
    %% Correct two factors rotate An such that A1n^T * A2n = diag(s)
    if param.correcttwofactors
        for n = 1:N-1
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
            
            %%
            Yh = 0;
            for p = 1:P
                Yh = Yh + tensor(Xp{p});
            end
            err = norm(Y - Yh)/normY;
            %Err = [Err err];
            %fprintf('Err2 %d\n',err)
            if ~isinf(errold2)
                done2 = abs(err-errold2) < errold2 * 1e-4;
            end
            errold2 = err;
            
        end
    end
    
    %     %%
    %     if param.rotatefactor
    %         Yh = 0;
    %         for p = 1:P
    %             Yh = Yh + tensor(Xp{p});
    %         end
    %
    %         Yr = Y - Yh;
    %
    %         for p = 1:P
    %             Yr = Yr + tensor(Xp{p});
    %             Yrot = ttm(Yr,Xp{p}.U,'t');
    %             for n = 1:N-1
    %                 Frot{n} = eye(R(p,n));
    %             end
    %             Trot = mtucker_als(Yrot,R(p,:),opt_hooi,Frot);
    %             Xp{p} = ttm(Trot,Xp{p}.U);
    %             Yr = Yr - tensor(Xp{p});
    %         end
    %     end
    
    %% Approximate error
    Yh = 0;
    for p = 1:P
        Yh = Yh + tensor(Xp{p});
    end
    err = norm(Y - Yh)/normY;
    Err = [Err err];
    fprintf('Err %d\n',err)
    if ~isinf(errold)
        done = (abs(err-errold) < errold * 1e-6) || (abs(err-errold) < 1e-6);
    end
    errold = err;
    
end


    function Xp = rotatefactor(Xp)
        % Rotate factors
        for p2 = 1:P
            for n =1:N-1
                [Xp{p2}.U{n},rr] = qr(Xp{p2}.U{n},0);
                Xp{p2}.core = ttm(Xp{p2}.core,rr,n);
            end
        end
        % Rotate factors to be orthogonal
        for n2 = 1:N-1
            [u,s,v] = svd(Xp{1}.U{n2}'*Xp{2}.U{n2},0);
            Xp{1}.U{n2} = Xp{1}.U{n2} * u;
            Xp{2}.U{n2} = Xp{2}.U{n2} * v;
            
            Xp{1}.core = ttm(Xp{1}.core,u',n2);
            Xp{2}.core = ttm(Xp{2}.core,v',n2);
        end
    end

    function Xp = updatecore(Y,Xp)
        % Update core tensors
        % update core
        W =[];
        for p2 = 1:P
            KRN = kron(Xp{p2}.U{N-1},Xp{p2}.U{N-2});
            for kn = N-3:-1:1
                KRN = kron(KRN,Xp{p2}.U{kn});
            end
            W = [W KRN];
        end
        G = W\reshape(Y.data,I(1)*I(2),I(3));
        
        G = mat2cell(G,prod(R,2),I(N));
        for p = 1:P
            Xp{p}.core = tensor(reshape(G{p},[R(p,:) I(N)]));
        end
        
    end

    function Xp = fastupdatecore(Y,Xp)
        % Call after rotate factors to be independent
        % update core
        %G = W\reshape(Y.data,I(1)*I(2),I(3));
        UtV = kron(sum(Xp{1}.U{2}.*Xp{2}.U{2}),sum(Xp{1}.U{1}.*Xp{2}.U{1}));
        
        G = [reshape(double(ttm(Y,Xp{1}.U,1:N-1,'t')),prod(R(1,:)),I(3))
            reshape(double(ttm(Y,Xp{2}.U,1:N-1,'t')),prod(R(2,:)),I(3))];
        
        % fast inverse has not yet implemented
        WtW = [eye(prod(R(1,:))) diag(UtV);
            diag(UtV) eye(prod(R(2,:)))];
        G = WtW\G;
        
        G = mat2cell(G,prod(R,2),I(N));
        for p = 1:P
            Xp{p}.core = tensor(reshape(G{p},[R(p,:) I(N)]));
        end
        
    end


    function Xp= alsupdate(Y,Xp)
        %--- Algorithm parameters
        comp='off';          % ='on' or ='off' to perform or not dimensionality reduction
        Tol1=1e-6;          % Tolerance
        MaxIt1=1;        % Max number of iterations
        Tol2=1e-4;          % tolerance in refinement stage (after decompression)
        MaxIt2=10;         % Max number of iterations in refinement stage
        Ninit=1;
        
        A_init = {Xp{1}.U{1} Xp{2}.U{1}};
        B_init = {Xp{1}.U{2} Xp{2}.U{2}};
        C_init = {double(Xp{1}.core) double(Xp{2}.core)};
        [A_est,B_est,C_est,phi,it1,it2,phi_als]=bcdLrMr_alsls(Y.data,...
            R(:,1)',R(:,2)','lsb',comp,Tol1,MaxIt1,Tol2,MaxIt2,Ninit,A_init,B_init,C_init);
                
        for p = 1:P
            Xp{p} = ttensor(tensor(C_est{p}),{A_est{p}, B_est{p}, eye(I(N))});
        end
    end
end