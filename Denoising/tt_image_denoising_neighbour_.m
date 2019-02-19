function Yhm = tt_image_denoising_neighbour_(Y,blk_size,neighb_range,sigma_noise, decomposition_method,shiftstep,get_rank,colorspace,Y0,im_name_)
%%%
% TENSORBOX, 2018

if nargin < 2
    blk_size = 8 * ones(1,2); % block size s
end
if nargin < 3
    neighb_range = 3;
end
if nargin <4
    % if noise level is not provided, it will be estimated from high
    % frequency coefficients 
    sigma_noise = noise_estimate(Y);
end
if nargin < 5
    % TT-algorithm to decompose a tensor 
    decomposition_method = 'tt_ascu';  %% tt_truncation, cpd, tucker
end
if nargin < 6
    shiftstep = 2;  
end

if nargin < 7
    get_rank = false;
end

if nargin < 8
    colorspace = 'rgb'; % 'opp
end

if nargin<10
    im_name_ = 'im';
end

step = ceil((blk_size(1)+neighb_range*2)/2);

shift_ix = unique([-step:shiftstep:step 0]);
Noshifts = numel(shift_ix)^2;
Yh = nan(numel(Y),Noshifts);
 
% Parameters for the denoising 
tt_options = tt_block_denoise_neighbour;
tt_options.noise_level = sigma_noise;
tt_options.neighb_range = neighb_range;
tt_options.block_size = blk_size;
tt_options.spatial_dictionary = 'dct'; % or predefined dictionary 
tt_options.decomposition_method = decomposition_method; % truncation or alternating update with noise given 
tt_options.get_rank = get_rank;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Change colorspace, compute the l2-norms of the new color channels
%%%%
if strcmp(colorspace,'opp')
    % transform the image to another color space
    [Y A l2normLumChrom] = function_rgb2LumChrom(Y, 'opp');
    % compute the scaling factor for the noise level
    
    % the noise level is recalculated after transforming the image    % 
    %
    %   |Y-Yn|_F^2 = sigma^2 * No_pixels * N_color_layers
    %  
    % N_color_layers = 3
    %
    %  Denote A the linear transform , 
    %    E = Y-Yn = sigma* sqrt(No_pixels) * I_N_layer
    % then 
    %   |(Y-Yn)*A|_F^2 = trace(A' * E'*E*A) 
    %         = sigma^2* No_pixels * trace(A'*A)
    %         = sigma^2* No_pixels * |A|_F^2
    N_layers = size(Y,3);
    sigma_noise = sigma_noise * norm(A(:))/sqrt(N_layers);
    tt_options.noise_level = sigma_noise;
end

%%
tt_options.process_name = im_name_;
fprocessed_ = sprintf('%s_block_processed',im_name_);

kshift = 0;
%% Shift the data to fast block processing 
for s1 = shift_ix
    for s2 = shift_ix
        
        
        kshift = kshift+1;
        fprintf('Shift %d/%d \n' , kshift,Noshifts)
        shift_ix2 = [s1 s2];
        
        fcurr_block_processed = sprintf('%s_%d.mat',fprocessed_,kshift);
        
        Yshift = circshift(Y,shift_ix2);        
        tt_options.process_id = kshift;
        
        %% Process blockes 8x8 and its neighbours
        tic
        Yx_shift = blockproc(Yshift,blk_size+2*neighb_range,...
            @(x) tt_block_denoise_neighbour(x,Yshift,tt_options),...
            'UseParallel',true);
        toc
        
        %             tic
        %             Yx_shift = blockproc(Yshift,blk_size+2*neighb_range+2,...
        %                 @(x) tt_block_denoise_neighbour_level2(x,Yshift,tt_options),...
        %                 'UseParallel',true);
        %             toc
        Yx_shifti = circshift(Yx_shift,-shift_ix2);
        Yh(:,kshift) = Yx_shifti(:);
        
        
        save(fcurr_block_processed,'kshift','Yx_shifti');

    end
end

%% Reconstruct the image 
Yhm = nanmedian(Yh,2);
Yhm = reshape(Yhm,size(Y));
Yhm(isnan(Yhm)) = Y(isnan(Yhm));

%% Convert back to RGB colorspace
if strcmp(colorspace,'opp')
    Yhm = function_LumChrom2rgb(Yhm, 'opp');
end
