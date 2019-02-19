% This file illustrates Rank-1 Rensor Deconvolution 
% which explains data through a convolutive model
%   
%    X = H_1 * (a_1 o b_1 o c_1)  + ... + H_R * (a_R o b_R o c_R)
%
% where X is a tensor of size I1 x I2 x I3, 
%       H_r are R patterns of size J1 x J2 x J3, 
% and   (a_r o b_r o c_r) are rank-1 tensors of size K1 x K2 x K3, 
% In = Jn + Kn -1.
%
% 
% TENSORBOX
%
% Phan Anh Huy, Aug 2014

clear all;

SzU = [15 16 17]; % size of rank-1 activating tensors
SzH = [2 2 2];   % Size of pattern tensors H: J1=J2=J3=J.
R = 4;         % no of patterns, i.e, the number of H_r
N = numel(SzU);
SNR = 30;       % Noise level (dB)

% Generate tensor X from random patterns H_r and rank-1 activating tensors
% a_r o b_r o c_r
[X,H0,U0] = gen_ts_conv(SzU,SzH,R);


%% Add Gaussian noise into the tensor X
sigma_noise = 10^(-SNR/20)*std(double(X(:)));
X = X + sigma_noise * randn(size(X));
X = tensor(X);

%% Rank-1 tensor deconvolution using the ALS algorithm
fprintf('ALS algorithm. \n The decomposition may run several times to get the best result.\n')

% Initialization for the tensor deconvolution
opts = ts_deconv_init;
opts.init = {'tedia' 'cpd1' 'cpd2'} ; % or {'tedia' 'cpd1' 'cpd2'} 
[Hn,Un] = ts_deconv_init(X,R,SzH,opts);

% Run ALS
opts = ts_deconv_rank1_als;
opts.maxiters = 2000;
opts.printitn = 1;
opts.tol = 1e-9;

tic;
[Hn,Un,output] = ts_deconv_rank1_als(X,Un,Hn, opts);
t_exec = toc;
% Hn are estimates of the tensors Hn
%
% Un{n} = [U{n,1}, ..., U{n,r}, ..., U{n,R}]: estimates of Un{r}

%% Evaluate performance 
% Compute the squared angular error
 
for r = 1:R
    u = cell(N,1);
    for n = 1:N
        u{n} = Un{n}(1:SzU(n),1,r);
    end
    
    Pr = ktensor(u);Pr = arrange(Pr);
    
    for s = 1:R
        [msae1,msae2,sae1,sae2] = SAE(U0{s},Pr.u(:));
        msae(r,s) = msae1;
    end
end

msae = min(msae);

fprintf('SAE (dB)   %s\n', sprintf('%.2f, ', -10*log10(msae)))
fprintf('Relative error %d\n',output.Error(end))
fprintf('Execution time %d seconds\n',t_exec)

return