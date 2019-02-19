% This file illustrates feature extraction for the MNIST handwritten
% digits using the Rank-1 Tensor Deconvolution.
%
%
% TENSORBOX v.2015.

%% Load the MNIST handwritten digit dataset for only two digits
clear all

compare_tdc_vs_cpd = false; % true to compare TDC with CPD with the same number of features
                            %
digits = [2 6];  % digits to be analyzed
Nosamples = 100; % number of images per digit
Nclasses = numel(digits);

X = [];TrueLabels = [];
for n = 1:Nclasses
    
    data1_st = sprintf('train%d',digits(n));
    try
        data1 = load('mnist_all.mat',data1_st);
    catch
        fprintf('Data file ''mnist_all.mat'' does not exist, or is not in the Matlab path.\n')
        fprintf('The mat file can be found at <a href="https://drive.google.com/file/d/0B6hgADLGQOkPaWVWTmJpWDM5MjQ/view"> https://drive.google.com/file/d/0B6hgADLGQOkPaWVWTmJpWDM5MjQ/view</a> \n')
    end
    data1 = data1.(data1_st);
    
    %% Concatenate tensors of digit1 and digit2 into one
    X = cat(1,X,reshape(double(data1(1:Nosamples,:)),[],28,28));
    
    % Define the true labels for the digits
    TrueLabels = [TrueLabels ; n*ones(Nosamples,1)];
end

X = permute(X,[3 2 1]); % data size 28 x 28 x 2*Nosamples

% Normalization and cropping digits
X = X/max(abs(double(X(:))));
X = X(3:end-2,3:end-2,:);       % trim images to size of 24 x 24

SzX = size(X);N = ndims(X);

%% Stage 1: Initialization for the rank-1 tensor deconvolution
R = 2;          % Two patterns will yield two features.
SzH = [8 8 1];  % Size of patterns, i.e. size of the core tensors H_r.
% When SzH = [1 1 1], the rank-1 tensor deconvolution becomes CPD of
% rank R.

init_opts = ts_deconv_init;
init_opts.init = 'cpd1';
[Hn,U] = ts_deconv_init(X,R,SzH,init_opts);

%% Stage 2:  run ALS algorithm for the rank-1 tensor deconvolution
Xcp = tensor(permute(double(X),[4 1 2 3]));

% Set parameters for the ALS algorithm
opts = ts_deconv_rank1_als;
opts.maxiters = 1000;
opts.tol = 1e-8;
opts.printitn = 1;
opts.orthomode = 0;
% opts.sparse = [1 2 3];

for kiter = 1:20
    tic;
    [Hn,U,outputk] = ts_deconv_rank1_als(Xcp,U,double(Hn),opts);
    t_chtd = toc;
    
    if kiter > 1
        if abs(output.Error(end)  - outputk.Error(end)) < opts.tol
            break
        end
        output = struct('Error',[output.Error outputk.Error],'NoIters',output.NoIters + outputk.NoIters);
    else
        output = outputk;
    end
end

figure(10);clf
loglog(output.Error);
xlabel('Iterations')
ylabel('Approximation Error')

%% Stage 3: Extract Features and perform clustering

% Feature matrices
Feat = U{N}(1:SzX(N),:);
Feat = bsxfun(@rdivide,Feat,sqrt(sum(Feat.^2)));

% KMEANS clustering
[idx,ctrs] = kmeans(Feat,Nclasses,'Distance','sqEuclidean','Replicates',500);
idx = bestMap(TrueLabels,idx);
%============= evaluate AC: accuracy ==============
acc  = length(find(TrueLabels == idx))/length(TrueLabels);


%% Stage 4: Visualize the results

% Construct the approximates
hX = squeeze(gen_ts_conv(Hn,U));

% Assess the relative approximation error
err = norm(tensor(hX - squeeze(X)))/norm(X(:));

fprintf('Tensor deconvolution: (R = %d) patterns of size %d x %d x %d\n',R,SzH);
fprintf('Approximation Error \t %.4f\nClustering Accuracy \t %.2f  \n',err, acc)

res_tdeconv = struct('ACC',acc,'Error',err);

figure(2);clf
montage(reshape(double(hX),[size(hX,1) size(hX,2) 1 size(hX,3)]),'Size',[SzX(N)/20 20])
title('Approximation using TDC')

% Construct basis images learnt by the tensor deconvolution.
% There are J basis images for each feature. So the total number of basis
% images are R *J.

Patt = [];
SzHn = size(Hn);
for r = 1:R
    Ur = cellfun(@(x) x(:,:,r), U,'uni',0);
    patr = ttm(tensor(Hn(:,:,:,:,r),[1 SzH]),Ur(1:2),[2 3]);
    patr = permute((patr),[2 3 1 4]);
    patr = double(patr);
    patr_rec = patr;
    
    Patt = [Patt double(reshape(patr_rec,SzX(1)*SzX(2),[]))];
end

Patt = bsxfun(@minus,Patt,mean(Patt));
Patt = bsxfun(@rdivide,Patt,sqrt(sum(Patt.^2)));

figure(3);clf
visual(Patt,SzX(1),R);
title('Basis patterns estimated by TDC')

% Scatter plot of feature 1 vs feature 2
figure(5)
h = gscatter(U{N}(1:SzX(N),1,1),U{N}(1:SzX(N),1,2),TrueLabels);

xlabel('Feature 1')
ylabel('Feature 2')
legend(h,arrayfun(@(x) num2str(x),digits,'uni',0))


%% Run this part to see performance using CPD of rank-R

if compare_tdc_vs_cpd
    opts = cp_fLMa;
    opts.init = {'dtld' 'nvec' 'random'};
    opts.maxiters = 5000;
    opts.printitn = 1;
    opts.maxboost = 1;
    
    [Pcp,out_cp] = cp_fLMa(tensor(squeeze(X)),R,opts);
    
    % Feature matrices
    Feat = Pcp.u{N}(1:SzX(N),:);
    
    % Feat = bsxfun(@minus,Feat,mean(Feat));
    Feat = bsxfun(@rdivide,Feat,sqrt(sum(Feat.^2)));
    
    % KMEANS clustering
    [idx,ctrs] = kmeans(Feat,Nclasses,'Distance','sqEuclidean','Replicates',500);
    idx = bestMap(TrueLabels,idx);
    %============= evaluate AC: accuracy ==============
    acc  = length(find(TrueLabels == idx))/length(TrueLabels);
    
    fprintf('CPD of rank R = %d\n',R);
    fprintf('Approximation Error \t %.4f\nClustering Accuracy \t %.2f  \n',1-out_cp.Fit(end), acc)
end