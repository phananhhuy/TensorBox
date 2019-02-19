function [rec_block,sub_topleft,sub_botright,Yout] = tt_block_denoise_neighbour(block_struct,Y,options)
% Low rank Approximation of the block at location "block_struct.location"
%   min_X \|Y - X \| <= noise_level^2
%  subject X has minimum rank.
% 
%
% TENSORBOX, 2018
%
param = inputParser;
param.KeepUnmatched = true;
param.addParameter('noise_level',0,@isscalar);
param.addParameter('block_size',[8 8]); 
param.addParameter('neighb_range',3,@isscalar); 
param.addParameter('spatial_dictionary','dct',@(x) ismember(x,{'dct' 'ksvd'})); 
param.addParameter('decomposition_method','tt_ascu',@(x) ismember(x,...
    {'tt_truncation' 'tt_ascu' 'ttmps_ascu' 'ttmps_adcu' 'ttmps_adcu1' 'ttmps_adcu2' ...
    'ttmps_atcu' 'ttmps_atcu1' 'ttmps_atcu2' 'ttmps_atcu3' 'cpd' 'tucker' 'brtf' ...
    'cpdepc'})); 
param.addParameter('get_rank',false,@islogical);

param.addParameter('mean_filtering',true,@islogical);
param.addParameter('spatial_filtering',true,@islogical);

param.addParameter('process_id',[],@isscalar);
param.addParameter('process_name','im',@ischar);
 
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
decomposition_method = param.decomposition_method;

mean_filtering = param.mean_filtering;
spatial_filtering = param.spatial_filtering;

%%
curr_xy = block_struct.location;
sz_area = 2*neighb_range+block_size; % size of block data

fprintf('Block (%d,%d)\n',block_struct.location(1),block_struct.location(2))


SzY = size(Y);
szY3 = size(Y,3); % for color image

subdata = block_struct.data;
subdata_size = block_struct.blockSize;
% if (blk_size(1) ~= blk_size(2)) ||  any(blk_size~= sz_area)
%     rec_block = nan(size(blk_data));
%     sub_topleft = [];
%     sub_botright =[];
% %     if options.get_rank
% %         rec_block = struct('data',rec_block,'rank',[]);
% %     end
%     return
% end

%% Get block_data 
% skip pixels which are on the border of the image, last rows and columns,
sub_topleft = [];sub_botright =[];
botright = curr_xy+block_size-1; 
if any(botright>SzY(1:2))
    if nargout > 3
        Yout = nan(size(Y));
    end
    rec_block = nan(size(block_size));
%     if options.get_rank
%         rec_block = struct('data',rec_block,'rank',[]);
%     end
    return
end

%% Get linear operator to generate tensor of size blk_sz x blk_sz x im_layers x neigh x neigh

% specify the search area from the point (cx,cy) in a subregion
% (topleft)-bottomright
sub_topleft = max(1,curr_xy-sz_area*5+1);
sub_botright = min(SzY(1:2),curr_xy+sz_area*5+1);
sub_Y = Y(sub_topleft(1):sub_botright(1),sub_topleft(2):sub_botright(2),:);
curr_xy_2 = curr_xy - sub_topleft+1; % the considered pixels become a point (cx2,cy2) in the sub-region

% [f_tensorization, data_size_f] = gen_tensorization_oper(block_size,neighb_range, szY3);
[f_tensorization, data_size_f] = gen_tensorization_oper_localregion(block_size, szY3,curr_xy_2,sub_Y);

%% Generate tensor from block data
data_sel = f_tensorization.A(sub_Y);

%% Mean filtering
if mean_filtering 
    mean_blk = mean(mean(data_sel,1),2);
    data_sel = bsxfun(@minus,data_sel,mean_blk);
end
 

%% Spatial filtering
% spatial_filtering = 1;
if spatial_filtering    
    [data_sel,Dict] = spatial_filter(data_sel,spatial_dictionary);
end

%% TT-decomposition of data blocksize*blksize x 3 x 3 x K
[tt_approx,apx_rank] = tensor_denoising(data_sel,decomposition_method,noise_level);

if options.get_rank
    
    if ~isempty(param.process_id)
        file_blkrank = sprintf('%s_denoise_blkrank_localregion_%s_%d.txt',param.process_name,decomposition_method,param.process_id);
        fid = fopen(file_blkrank,'a+');
        fprintf(fid,'%d,%d, %d, %s\n',curr_xy(1),curr_xy(2),param.process_id,sprintf('%d, ',apx_rank));
    
    else
        file_blkrank = sprintf('%s_denoise_blkrank_localregion.txt',param.process_name);
        fid = fopen(file_blkrank,'a+');
        fprintf(fid,'%d,%d, %s\n',curr_xy(1),curr_xy(2),sprintf('%d, ',apx_rank));
   
    end
    fclose(fid);
end

tt_approx = double(reshape(full(tt_approx),size(tt_approx)));

%%
if spatial_filtering
    tt_approx = reshape(Dict*reshape(tt_approx,size(tt_approx,1)*size(tt_approx,2),[]),...
        data_size_f);
end

if mean_filtering
    tt_approx = bsxfun(@plus,tt_approx,mean_blk);
end
tt_approx = reshape(tt_approx,data_size_f);

%% Estimate pixels which correspond to the "tt_approx"
% using invert process of Tensorization
subY_x = f_tensorization.At(tt_approx);

%%
rec_block = subY_x;


if nargout > 3
    Yout = nan(size(Y));
    Yout(sub_topleft(1):sub_botright(1),sub_topleft(2):sub_botright(2),:) = subY_x;
end
    

% end of the main process 


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


    function [tt_approx,apxrank] = tensor_denoising(data_sel,decomposition_method,noise_level)
        warning('off');
        
        switch decomposition_method
            case 'tt_truncation'
%                 C = 1.05;
                C = 1.01;
                accuracy = C^2*noise_level^2*numel(data_sel);
                
                tt_1 = tt_tensor_denoise(data_sel,accuracy);
                % err0 = norm(data_sel(:) - full(tt_1))^2/norm(data_sel(:))^2;
                tt_approx = tt_1;
                apxrank = rank(tt_approx)'; 
                
            case 'tt_ascu'
%                 C = 1.05; % might be high if noise 0 dB
                C = 1.01; % might be high if noise 0 dB

                accuracy = C^2*noise_level^2*numel(data_sel);
                
                tt_1 = tt_tensor_denoise(data_sel,accuracy);
                % err0 = norm(data_sel(:) - full(tt_1))^2/norm(data_sel(:))^2;
                tt_approx = tt_1;
                
                
                % 2a ACULRO to data
                rankR = [];
                opts = tt_ascu;
                opts.compression = 0;
                opts.maxiters = 200;%maxiters;
                opts.tol = 1e-8;
                opts.init = tt_1;
                %opts.noise_level = sqrt(2) * noise_level^2*numel(data_sel);% * prod(size(Y));
                opts.noise_level = accuracy;
                opts.printitn = 0;
                
                [tt_approx,out2a] = tt_ascu(data_sel,rankR,opts);
                
                apxrank = rank(tt_approx)'; 
                
             case 'ttmps_ascu'
%                 C = 1.05; % might be high if noise 0 dB
                %C = 1.01; % might be high if noise 0 dB
                C = 1;
                
                accuracy = C^2*noise_level^2*numel(data_sel);
                
                tt_1 = tt_tensor_denoise(data_sel,accuracy);
                % err0 = norm(data_sel(:) - full(tt_1))^2/norm(data_sel(:))^2;
                tt_approx = tt_1;
                
                
                % 2a ACULRO to data
                rankR = [];
                opts = ttmps_a2cu;
                opts.compression = 0;
                opts.maxiters = 200;%maxiters;
                opts.tol = 1e-8;
                opts.init = tt_1;
                %opts.noise_level = sqrt(2) * noise_level^2*numel(data_sel);% * prod(size(Y));
                opts.noise_level = accuracy;
                opts.core_step = 2;
                opts.rankadjust = 2;
                opts.printitn = 0;
                
                [tt_approx,out2a] = ttmps_ascu(data_sel,rankR,opts);
                
                apxrank = tt_approx.rank; 
                
            case {'ttmps_adcu' 'ttmps_adcu1' 'ttmps_adcu2'}
                
                step = str2double(decomposition_method(end));
                if isnan(step)
                    step = 2;
                end

%                 C = 1.05; % might be high if noise 0 dB
                C = 1.01; % might be high if noise 0 dB
                
                accuracy = C^2*noise_level^2*numel(data_sel);
                
                tt_1 = tt_tensor_denoise(data_sel,accuracy);
                % err0 = norm(data_sel(:) - full(tt_1))^2/norm(data_sel(:))^2;
                tt_approx = tt_1;
                
                
                % 2a ACULRO to data
                rankR = [];
                opts = ttmps_a2cu;
                opts.compression = 0;
                opts.maxiters = 200;%maxiters;
                opts.tol = 1e-8;
                opts.init = tt_1;
                %opts.noise_level = sqrt(2) * noise_level^2*numel(data_sel);% * prod(size(Y));
                opts.noise_level = accuracy;
                %opts.core_step = 2;
                opts.core_step = step;
                opts.printitn = 0;
                
                [tt_approx,out2a] = ttmps_a2cu(data_sel,rankR,opts);
                
                apxrank = tt_approx.rank; 
                
            case {'ttmps_atcu' 'ttmps_atcu1' 'ttmps_atcu2' 'ttmps_atcu3'}
                step = str2double(decomposition_method(end));
                if isnan(step)
                    step = 2;
                end
                
                %                 C = 1.05; % might be high if noise 0 dB
                C = 1.01; % might be high if noise 0 dB
                
                accuracy = C^2*noise_level^2*numel(data_sel);
                
                tt_1 = tt_tensor_denoise(data_sel,accuracy);
                % err0 = norm(data_sel(:) - full(tt_1))^2/norm(data_sel(:))^2;
                tt_approx = tt_1;
                
                
                % 2a ACULRO to data
                rankR = [];
                opts = ttmps_a2cu;
                opts.compression = 0;
                opts.maxiters = 200;%maxiters;
                opts.tol = 1e-8;
                opts.init = tt_1;
                %opts.noise_level = sqrt(2) * noise_level^2*numel(data_sel);% * prod(size(Y));
                opts.noise_level = accuracy;
                opts.core_step = step;
                opts.printitn = 0;
                
                [tt_approx,out2a] = ttmps_a3cu(data_sel,rankR,opts);
                apxrank = tt_approx.rank;
                
            case 'cpdepc'
                
                %  CPD decompition
                C = 1.01; % might be high if noise 0 dB
                accuracy = C*noise_level*sqrt(numel(data_sel));
                
                opt_cp = cp_fLMa;
                opt_cp.init = [repmat({'rand'},1,20) 'nvec'];
                opt_cp.tol = 1e-8;
                opt_cp.maxiters = 1000;
                opt_cp.maxboost = 0;
                opt_cp.printitn = 0;
                
                apxrank = [];
                
                doing_estimation = true;
                Rcp = 8;
                best_result = [];
                
                data_ = tensor(data_sel);
                while doing_estimation
                    
                    [tt_approx] = cp_nc_sqp(data_,Rcp,accuracy,opt_cp);
                    tt_approx = normalize(tt_approx);
                    tt_approx = arrange(tt_approx);
                    
                    err_cp = norm(data_ - full(tt_approx));
                    norm_lda = norm(tt_approx.lambda(:));
                    
                    %%
                    if err_cp <= accuracy + 1e-5
                        % the estimated result seems to be good
                        if isempty(best_result) || (Rcp < best_result(1))
                            best_tt_approx = tt_approx;
                            best_result = [Rcp err_cp norm_lda];
                        end
                        
                        if (Rcp > 1)   % try the estimation with a lower rank
                            Rcp_new = Rcp-1;
                        else
                            doing_estimation = false;
                        end
                    else
                        Rcp_new = Rcp+1;
                    end
                    
                    apxrank = [apxrank ; [Rcp  err_cp norm_lda]];
                    if any(apxrank(:,1) == Rcp_new)
                        doing_estimation = false;
                    end
                    Rcp = Rcp_new;
                end

                tt_approx = best_tt_approx;
                
                %%
            case 'cpd'                
                
                Rcp = 8;

                opt_cp = cp_fLMa;
                opt_cp.init = [repmat({'rand'},1,20) 'nvec'];
                
                opt_cp.tol = 1e-8;
                opt_cp.maxiters = 1000;
                opt_cp.maxboost = 0;
                opt_cp.printitn = 0;
                
                while Rcp >=1 
                    try
                        
                        tt_approx = cp_fastals(tensor(data_sel),Rcp,opt_cp);
                        
                        if strcmp(decomposition_method,'cpdepc') && (Rcp>1)
                            % RUN ANC + fLM
                            %nc_alg = @cp_anc;
                            nc_alg = @cp_nc_sqp;
                            cp_alg = @cp_fLMa;
                            max_rank1norm = (10^2*Rcp);
                            epctol = 1e-7;epc_maxiters= 1000;
                            [tt_apprm ox,output_cpepc] = exec_bcp(tensor(data_sel),tt_approx,cp_alg,nc_alg,3,max_rank1norm,epctol,epc_maxiters);
                        end
                        break
                    catch 
                        Rcp = Rcp-1;
                    end
                end 
                apxrank = size(tt_approx.U{1},2);

            case 'brtf'
                fprintf('------Bayesian Robust Tensor Factorization---------- \n');
                % This method will not work if the data is relatively small
                % values ~~ 0.2. 
                % Scaling the data up by a factor of 1000 may work for some
                % cases.
                mx = max(abs(data_sel(:)));
                data_sel(abs(data_sel(:))<mx*1e-4) = 0;
                scaling = 1e8/mx;
                [model] = BayesRCP(data_sel*scaling, 'init', 'ml', ...
                    'maxRank',5*max(size(data_sel)), 'maxiters', 50, 'initVar',scaling*noise_level^2,...
                    'updateHyper', 'on','tol', 1e-5, 'dimRed', 1, 'verbose', 0);
                 
                 tt_approx = 1/scaling*ktensor(model.Z);
                 apxrank = size(tt_approx.U{1},2);

            case 'tucker' %  TUCKER denoising
                C = 1.01;
                
                opt_td.init = 'nvecs';
                opt_td.noiselevel = C*noise_level;
                opts.printitn = 0;
                
                tt_approx = tucker_als_dn(tensor(data_sel),size(data_sel),opt_td);
                
                apxrank = size(tt_approx.core);
        end
        
    end
end

%%

%%
function [f_tensorization, data_size_f] = gen_tensorization_oper_localregion(block_size, no_image_layers,curr_xy,Yn)
% for neighbour region of windows defined by block_size
%
% Y can be a sub data of the whole one.
% creat a binary mask for the current pixel
szY = size(Yn); 
sz_area = szY(1:2);

mask = false(szY(1),szY(2)); 
mask(curr_xy(1),curr_xy(2)) = true;
% Yn_bw = rgb2gray(Yn);
Yn_bw = Yn(:,:,1); % Lab color space

W = graydiffweight(Yn_bw, mask, 'GrayDifferenceCutoff', 25);

thresh = 0.08;
[BW, D] = imsegfmm(W, mask, thresh);
[neighb_ix,neighb_iy] = find(BW); % pixels are in the same segment 


% omit the points on the border.
botright = [neighb_ix neighb_iy]+block_size;
f_points = any(botright>sz_area,2);
neighb_ix(f_points) = [];
neighb_iy(f_points) = [];
maxNo_neighbours = 50;
if numel(neighb_ix) > maxNo_neighbours
    neighb_idx = sub2ind(sz_area,neighb_ix,neighb_iy);
    [Dsel,reordix] = sort(D(neighb_idx),'ascend');
    reordix = reordix(1:50);
    neighb_ix = neighb_ix(reordix);
    neighb_iy = neighb_iy(reordix);
    neighb_idx = neighb_idx(reordix);
    %find(sqrt(sum(([neighb_ix,neighb_iy]-curr_xy).^2,2)) < 20)
end
%sz_area= block_size + 2*neighb_range;
blk_ids = ind2sub_full(block_size,1:prod(block_size)); % meshgrid(1:blksize,1:blksize)

% pixels in the selected area indicated by the binary image BW [topleft_xy to botright_xy]

% [neighb_ix,neighb_iy]= meshgrid((1:2*neighb_range+1)',...
%     (1:2*neighb_range+1)');

pix_ix = bsxfun(@plus,neighb_ix(:)',blk_ids(:,1)-1); %row indices in area of size sz_area x sz_area
pix_iy = bsxfun(@plus,neighb_iy(:)',blk_ids(:,2)-1); %column indices in area of size sz_area x sz_area

pix_lind = sub2ind(sz_area,pix_ix(:),pix_iy(:));  % linear indices of selected pixels

% linear operator generates data from the sub_Y
% size blk_sz x blk_sz x neighbour width x neighbour width
Asel = speye(prod(sz_area),prod(sz_area));
Asel = Asel(pix_lind,:);
scaleA = diag((Asel'*Asel));

no_samples = size(neighb_ix,1);  % number of pixels are used to generate the tensor
no_selpoints = size(pix_lind,1);
data_size_f = [block_size no_samples no_image_layers];

f_tensorization.A = @(x) reshape(Asel*reshape(x,[],size(x,3)),data_size_f);
f_tensorization.At = @(x) reshape(bsxfun(@rdivide,Asel'*reshape(x,no_selpoints,[]),scaleA),[sz_area no_image_layers]);
end

%%
function [f_tensorization, data_size_f] = gen_tensorization_oper(block_size,neighb_range, no_image_layers)
% for neighbour region of windows defined by block_size
% f_tensorization.A(X) : forward operator generates a tensor for an
%                        observed location from its neighbour
%
% f_tensorization.At(X) : backward operator
%

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

f_tensorization.A = @(x) reshape(Asel*reshape(x,[],size(x,3)),data_size_f);
f_tensorization.At = @(x) reshape(bsxfun(@rdivide,Asel'*reshape(x,no_samples,[]),scaleA),[sz_area no_image_layers]);
end
