function rec_block = tt_block_denoise_neighbour(block_struct,Y,options)
%
% TENSORBOX, 2018
%
param = inputParser;
param.KeepUnmatched = true;
param.addParameter('noise_level',0,@isscalar);
param.addParameter('block_size',[8 8]); 
param.addParameter('neighb_range',3,@isscalar); 
param.addParameter('spatial_dictionary','dct',@(x) ismember(x,{'dct' 'ksvd'})); 

 
if nargin == 0
    options = struct;
end
param.parse(options);
param = param.Results;
if nargin == 0
    rec_block = param;
    return
end


noise_level = param.noise_level;
block_size = param.block_size;
neighb_range = param.neighb_range;
spatial_dictionary = param.spatial_dictionary;



%%
curr_xy = block_struct.location;
sz_area = 2*neighb_range+block_size; % size of block data

fprintf('Block (%d,%d)\n',curr_xy(1),curr_xy(2))


SzY = size(Y);
szY3 = size(Y,3); % for color image

blk_data = block_struct.data;

blk_size = block_struct.blockSize;
if (blk_size(1) ~= blk_size(2)) ||  any(blk_size< sz_area)
    rec_block = nan(size(blk_data));
    return
end

%% Get block_data
sub_Y = zeros([prod(sz_area)*szY3 (1+blk_size-sz_area)]);
for k1 = 0:blk_size(1)-sz_area(1)
    for k2 = 0:blk_size(2)-sz_area(2)
        
        % get new block of size sz_area from [k1,k2]
        obs_xy = [k1 k2]+1;        
        botright = obs_xy+sz_area-1;        
        sub_Y(:,k1+1,k2+1) = reshape(blk_data(obs_xy(1):botright(1),obs_xy(2):botright(2),:),[],1);
    end
end
sub_Y = reshape(sub_Y,[sz_area szY3 prod(1+blk_size-sz_area)]);

%% Get linear operator to generate tensor of size blk_sz x blk_sz x im_layers x neigh x neigh

[f_tensorization, data_size_f] = gen_tensorization_oper(block_size,neighb_range, szY3);

%% Generate tensor from block data
data_sel = f_tensorization.A(sub_Y);  % blocksize1 x blocksize2 x neighwidth x neighwidth x number_layers x prod(1+blk_size-sz_area)

data_size_f_xt = [data_size_f prod(blk_size-sz_area+1)];
data_sel = reshape(data_sel,data_size_f_xt);

%% Mean filtering
mean_blk = mean(mean(data_sel,1),2);
data_sel = bsxfun(@minus,data_sel,mean_blk);

%% Spatial filtering
spatial_filtering = 1;
if spatial_filtering    
    [data_sel,Dict] = spatial_filter(data_sel,spatial_dictionary);
end

%% TT-decomposition of data blocksize*blksize x 3 x 3 x K
decomposition_method = 'tt'; % 'tt' cpd' 'tucker'
data_sz = size(data_sel);
data_sel = reshape(data_sel,[data_sz(1)*data_sz(2) data_sz(3:end)]);

tt_approx = tensor_denoising(data_sel,decomposition_method,noise_level);

tt_approx = double(reshape(full(tt_approx),size(tt_approx)));
tt_approx = reshape(tt_approx,data_sz);

%%
if spatial_filtering
    tt_approx = reshape(Dict*reshape(tt_approx,size(tt_approx,1)*size(tt_approx,2),[]),...
        data_size_f_xt);
end

tt_approx = bsxfun(@plus,tt_approx,mean_blk);
tt_approx = reshape(tt_approx,data_size_f_xt);

%% Estimate pixels which correspond to the "tt_approx"
% using invert process of Tensorization
subY_x = f_tensorization.At(tt_approx);

%% Recon
if szY3>1
    subY_x = reshape(subY_x,[sz_area szY3 (blk_size-sz_area+1)]);
else
    subY_x = reshape(subY_x,[sz_area (blk_size-sz_area+1)]);
end

rec_block = zeros(size(blk_data));
rec_cnt = zeros(size(blk_data));
for k1 = 1:blk_size(1)-sz_area(1)+1
    for k2 = 1:blk_size(2)-sz_area(2)+1  
        obs_xy =[k1 k2];
        botright = obs_xy+sz_area-1;        
        if szY3>1
            rec_block(obs_xy(1):botright(1),obs_xy(2):botright(2),:) = ...
                rec_block(obs_xy(1):botright(1),obs_xy(2):botright(2),:) + subY_x(:,:,:,k1,k2);
        else
            rec_block(obs_xy(1):botright(1),obs_xy(2):botright(2)) = ...
                rec_block(obs_xy(1):botright(1),obs_xy(2):botright(2)) + subY_x(:,:,k1,k2);
        end
        rec_cnt(obs_xy(1):botright(1),obs_xy(2):botright(2),:) = ...
            rec_cnt(obs_xy(1):botright(1),obs_xy(2):botright(2),:) +1;
    end
end

%%
rec_block = rec_block./rec_cnt;


    function  [data_sel,Dict] = spatial_filter(data_sel,Dict)
        
        switch Dict
            case 'dct'
                ldic = dctmtx(block_size(1));
                rdic = dctmtx(block_size(2));
                Dict = kron(rdic,ldic);
                
                data_sel = reshape(Dict'*reshape(data_sel,size(data_sel,1)*size(data_sel,2),[]),...
                    size(data_sel));
            
            case 'ksvd'
                param.K = block_size(1);
                param.preserveDCAtom = 0;
                param.numIteration = 10;
                param.errorFlag =1 ;
                C = 1.15;
                param.errorGoal = noise_level*C;
                
                ldic = dctmtx(block_size(1));
                param.initialDictionary = ldic(:,1:param.K );
                param.InitializationMethod =  'GivenMatrix';
                
                [dict_left] = KSVD(double(tenmat(data_sel,1)),param);
                
                ldic = dctmtx(block_size(2));
                param.initialDictionary = ldic(:,1:param.K );
                [dict_right] = KSVD(double(tenmat(data_sel,2)),param);
                
                Dict = kron(dict_right,dict_left);
                
                sz = size(data_sel);
                data_sel = reshape(Dict'*reshape(data_sel,size(data_sel,1)*size(data_sel,2),[]),...
                    [K K sz(3:end)]);
        end        
    end


    function tt_approx = tensor_denoising(data_sel,decomposition_method,noise_level)
        
        switch decomposition_method
            case 'tt'
                C = 1.05;
                
                accuracy = C^2*noise_level^2*numel(data_sel);
                
                tt_1 = tt_tensor_denoise(data_sel,accuracy);
                % err0 = norm(data_sel(:) - full(tt_1))^2/norm(data_sel(:))^2;
                tt_approx = tt_1;
                
                
                % 2a ACULRO to data
                rankR = [];
                opts = tt_ascu;
                opts.compression = 0;
                opts.maxiters = 200;%maxiters;
                opts.tol = 1e-6;
                opts.init = tt_1;
                %opts.noise_level = sqrt(2) * noise_level^2*numel(data_sel);% * prod(size(Y));
                opts.noise_level = accuracy;
                
                [tt_approx,out2a] = tt_ascu(data_sel,rankR,opts);
                
                
                
                
            case 'cpd'
                %  CPD decompition
                
                Rcp = 20;
                opt_cp = cp_fLMa;
                opt_cp.init = 'nvecs';
                opt_cp.to = 1e-8;
                opt_cp.maxiters = 50;
                opt_cp.maxboost = 0;
                tt_approx = cp_fastals(tensor(data_sel),Rcp,opt_cp);
                                
            case 'tucker' %  TUCKER denoising
                C = 1.01;
                
                opt_td.init = 'nvecs';
                opt_td.noiselevel = C*noise_level;
                
                tt_approx = tucker_als_dn(tensor(data_sel),size(data_sel),opt_td);
        end
        
    end
end



function [f_tensorization, data_size_f] = gen_tensorization_oper(block_size,neighb_range, no_image_layers)

sz_area= block_size + 2*neighb_range;
blk_ids = ind2sub_full(block_size,1:prod(block_size)); % meshgrid(1:blksize,1:blksize)

% pixels in area [topleft_xy to botright_xy]

[neighb_ix,neighb_iy]= meshgrid((1:2*neighb_range+1)',...
    (1:2*neighb_range+1)');

pix_ix = bsxfun(@plus,neighb_ix(:)',blk_ids(:,1)-1); %row indices in area of size sz_area x sz_area
pix_iy = bsxfun(@plus,neighb_iy(:)',blk_ids(:,2)-1); %column indices in area of size sz_area x sz_area

pix_lind = sub2ind(sz_area,pix_ix(:),pix_iy(:));  % linear indices of selected pixels

% linear operator generates data from the sub_Y
% size blk_sz x blk_sz x neigh x neigh
Asel = speye(prod(sz_area),prod(sz_area));
Asel = Asel(pix_lind,:);
scaleA = diag((Asel'*Asel));

data_size_f = [block_size neighb_range*2+1 neighb_range*2+1 no_image_layers];
no_samples = size(pix_lind,1);  % number of pixels are used to generate the tensor

f_tensorization.A = @(x) reshape(Asel*reshape(x,size(x,1)*size(x,2),[]),[data_size_f sizex(x,4:ndims(x))]);

% input x is a tensor of size 
% blocksize x blocksize x neighwidth x neighwidth x no_layers x K1 x K2 ...
% 
f_tensorization.At = @(x) reshape(bsxfun(@rdivide,Asel'*reshape(x,no_samples,[]),scaleA),[sz_area sizex(x,5:ndims(x))]);
end

function sz = sizex(x,dims)
sz = size(x);
try 
    sz = sz(dims);
catch 
    sz = 1;
end
end

