function [criball,cribf,H] = cribBCDLoR(U,LR,rankRdim)
% Cramer-Rao- induced lower bound ofr rank-LoM structured CPD
% The noise  is assumed to be i.i.d. Gaussian of a variance sigma^2, 
% the output CRB is proportional to sigma^2, and without any loss in generality it is assumed that
%  is equal to 1.
%
%
% 
% Ref: P. Tichavsk?y, A. H. Phan, and Z. Koldovsk?y, ?Cram?er-Rao-Induced Bounds
% for CANDECOMP/PARAFAC tensor decomposition,? ArXiv e-prints, Sept. 2012,
% http://arxiv.org/abs/1209.3215. 
%

  
I = cellfun(@(x) size(x,1),U);I = I(:)';
In = I;
N = numel(U);

rankLdim = setdiff(1:N,rankRdim);
 
%% Method 4 :  COnstruct Hessian 
R = sum(prod(LR,1)); % total rank structured CPD
NoPat = size(LR,2);

Eleft = [];
Eright = [];
for p = 1:NoPat
    Eleft = blkdiag(Eleft,kron(eye(LR(1,p)),ones(1,LR(2,p))));
    Eright = blkdiag(Eright,kron(ones(1,LR(1,p)),eye(LR(2,p))));
end

[Prr,Ptr] = per_vectrans(R,R);

E = cell(N,1);
E(rankLdim) = {Eleft};
E(rankRdim) = {Eright};
 
for n = 1:N
    if ismember(n,rankLdim)
        A{n} = U{n} * Eleft;
    else
        A{n} = U{n} * Eright;
    end
end

AtA = zeros(R,R,N);
for n = 1: N
    AtA(:,:,n) = A{n}'*A{n};
end

G =[];
for n = 1:N
    gnn = prod(AtA(:,:,setdiff(1:N,[n])),3);
    if ismember(n,rankRdim)
        gnn = Eright * gnn * Eright';
    else
        gnn = Eleft * gnn * Eleft';
    end
    G = blkdiag(G,kron(gnn, eye(In(n))));
end

Z = [];
for n = 1: N
    Zn = kron(eye(size(E{n},1)),U{n});
    Z = blkdiag(Z,Zn);
end


% cR = cumsum([0 R^2*ones(1,N)]);
% K = zeros(N*R^2);

Rf = [sum(LR(1,:))^2*ones(1,numel(rankLdim)) sum(LR(2,:))^2*ones(1,numel(rankRdim))];
Rf([rankLdim rankRdim]) = Rf;
cR = cumsum([0 Rf]);
% cR([rankLdim rank1dim])
P = numel(rankLdim);
K = zeros(P*sum(LR(1,:))^2 + (N-P) * sum(LR(2,:))^2);
% cR = cumsum([0 R^2*ones(1,P) Rs*R*ones(1,N-P)]);
% K = zeros(P*R^2 + (N-P) * Rs^2);
for n = 1:N-1
    for m = n+1:N
        gmn = prod(AtA(:,:,setdiff(1:N,[n,m])),3);
        Knm = Prr*diag(gmn(:));
       
        
        Knm = kron(E{n},E{n}) * Knm * kron(E{m}',E{m}');
        
        
        K(cR(n)+1:cR(n+1),cR(m)+1:cR(m+1)) =  Knm;
        K(cR(m)+1:cR(m+1),cR(n)+1:cR(n+1)) =  Knm';
    end
end

H = G + Z * K * Z'; % Hessian

%%
len=size(H,1);
Ind=1:len;

subrank = [sum(LR(1,:))*ones(1,P)  sum(LR(2,:))*ones(1,N-P)];
subrank([rankLdim rankRdim]) = subrank;
    

cIR = cumsum(I.*subrank); % need to fix order here
cIR = [0 cIR];


Hi = inv(H+1e-8*eye(size(H,1)));
criball = nan(N,max(sum(LR,2)));
for n = 1:N
    for k = 1:size(U{n},2)
        Pa1=eye(I(n))-U{n}(:,k)*U{n}(:,k)'/sum(U{n}(:,k).^2);
        k1=cIR(n)+(k-1)*I(n);
        criball(n,k)=sum(diag(Pa1*Hi(k1+1:k1+I(n),k1+1:k1+I(n))))/sum(U{n}(:,k).^2);
    end
end
cribf = [];

% %% INVERSE Hessian
% idx = [];
% for n = 2:N
%     idx =[idx cIR(n)+1:I(n):cIR(n+1)];
% end
% Ind(idx) = [];
% Hi2=inv(H(Ind,Ind));
% 
% %%
% %crib=sum(diag(Pa1*Hi2(1:Ia,1:Ia)))/sum(A(:,1).^2);
% criball = zeros(1,1);
% n = 1;
% for k=1:1
%     Pa1=eye(I(n))-U{n}(:,k)*U{n}(:,k)'/sum(U{n}(:,k).^2);
%     k1=cIR(n)+(k-1)*I(n);
%     criball(1,k)=sum(diag(Pa1*Hi2(k1+1:k1+I(n),k1+1:k1+I(n))))/sum(U{n}(:,k).^2);
% end
