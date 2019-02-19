function [ns,bestord,mincost,Allcomb] = cost_als(I)
% I should be in ascending order
%
% Phan Anh Huy , 2013
N = numel(I);% N should be smaller than 15
% Compute total cost of the fastALS
%ns
Allcomb= [];
AllTn = [];
for ns = ceil(N/2):N-1
    % Jn = cumprod(I); Kn = [Jn(end) Jn(end)./Jn(1:end-1)];
    allcomb = nchoosek(1:N,ns);
    temp = zeros(size(allcomb,1),N-ns);
    for k = 1:size(allcomb,1)
        temp(k,:) = setdiff(1:N,allcomb(k,:));
    end
    allcomb = [allcomb temp];
    
    Jn = cumprod(I(allcomb),2);
    Kn = [Jn(:,end) bsxfun(@rdivide,Jn(:,end),Jn(:,1:end-1))];
    
    %%
    Tn = 0;
    for n = [ns:-1:1 ns+1:N]
        if (n == ns)
            Mn = sum(Jn(:,2:n-1),2) + min(Jn(:,n),Kn(:,n)) + ...
                 sum(Jn(:,n+2:N),2)./Jn(:,n);
        elseif n== ns+1
            Mn = sum(Jn(:,2:n-1),2) + (ns<N-1)*min(Jn(:,n),Kn(:,n)) + ...
                sum(Jn(:,n+2:N),2)./Jn(:,n);
        elseif n< ns
            Mn = sum(Jn(:,2:n+1),2);
        elseif n>ns+1
            Mn = Kn(:,n-1) + (n~= N) * Kn(:,n) + sum(Jn(:,n+2:N),2)./Jn(:,n);
        end
        Tn = Tn + Mn;
    end
    if nargout >= 4
        Allcomb = [Allcomb; allcomb Tn(:) ns*ones(size(allcomb,1),1) ];
    else
        [mincost,idbest] = min(Tn(:));
        bestord = allcomb(idbest,1:N);
        Allcomb = [Allcomb; bestord mincost ns];
    end
end

[mincost,idbest] = min(Allcomb(:,N+1));
bestord = Allcomb(idbest,1:N);
ns = Allcomb(idbest,end);