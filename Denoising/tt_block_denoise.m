function rec_block = tt_block_denoise(block_struct,Y,noise_level)
%
% TENSORBOX, 2018

%%
%cprintf('r_','(%d,%d)',i,j)
% select block
fprintf('(%d,%d)\n',block_struct.location(1),block_struct.location(2))
%     rec_block = nan(block_struct.blockSize);

% return
blk_data = block_struct.data;

blk_size_x = block_struct.blockSize;
if (blk_size_x(1) ~= blk_size_x(2)) ||  any(blk_size_x~= 24)
    rec_block = nan(size(blk_data));
    return
end
blk_size= blk_size_x/3;
% select blocks which are similar to the center block, i.e., (2,2)
ctr_blk = blk_data(blk_size(1)+1:2*blk_size(1),blk_size(2)+1:2*blk_size(2),:);


%% select blocks which are similar to the current block
step = [4 4];
Noshifts = prod(2*step+1);
cnt = 1;
sel_blocks = zeros([blk_size*3 size(ctr_blk,3)]);
xc_sel = [];
for kshift = 1:Noshifts
    shift_ix = ind2sub_full(2*step+1,kshift);
    shift_ix = shift_ix - step - 1;
    Ys = circshift(Y,shift_ix);
    
    xfun = @(block_struct,ref) corr2(block_struct.data,ref);
    xc_blk = blockproc(Ys,blk_size,@(x) xfun(x,ctr_blk));
    [xc_blks,ii] = sort(abs(xc_blk(:)),'descend');
    if xc_blks(1) == 1
        xc_blks(1) = [];
        ii(1) = [];
    end
    
    ii = ii((xc_blks>.3));
    sub_ii = ind2sub_full(size(xc_blk),ii);
    sub_ii = blk_size(1)*(sub_ii-1)+1;
    
    if ~isempty(ii)
        for ki = 1:numel(ii)
            try
                sel_blocks(:,:,cnt) = Ys(sub_ii(ki,1)-blk_size(1):sub_ii(ki,1)+blk_size(1)*2-1,...
                    sub_ii(ki,2)-blk_size(2):sub_ii(ki,2)+blk_size(2)*2-1,:);
                
                xc_sel(cnt) = xc_blks(ki);
                cnt = cnt+1;
            catch
                %                         sub_ii(ki,:)
            end
        end
    else
        % %                 111
    end
end
size(sel_blocks)

%% REshape the selected blocks
if cnt > 1
    data_sel = cat(3,blk_data,sel_blocks);
else
    data_sel = blk_data;
end
data_sel = permute(reshape(data_sel,blk_size(1),3,blk_size(2),3,[]),[1 3 2 4 5]);
data_sel = reshape(data_sel,blk_size(1),blk_size(2),3,3,[]);

% mean_blk = mean(mean(data_sel,1),2);
% data_sel = bsxfun(@minus,data_sel,mean_blk);

mean_blk = mean(mean(mean(data_sel,5),4),3);
data_sel = bsxfun(@minus,data_sel,mean_blk);

%% TT-decomposition
%         tt_1 = tt_tensor(data_sel,1e-3);

%noise_level = sigma_noise^2*numel(data_sel)*(ndims(data_sel)-1);
% noise_level = sigma_noise^2*numel(data_sel)*sqrt(2)*factx;
%tt_1 = tt_tensor(data_sel,noise_level);

accuracy = noise_level^2*numel(data_sel)*sqrt(2);

tt_1 = tt_tensor_denoise(data_sel,accuracy);

% err0 = norm(data_sel(:) - full(tt_1))^2/norm(data_sel(:))^2;
% %
% err0

tt_2 = tt_1;

%         fig = figure(1);
%         clf; hold on
%         clear h
%         h(1) = plot(1,err0,'o');
%         set(h(1),'markersize',18)
%

%         % 2a ACULRO to data
%         rankR = [];
%         opts = tt_aculro;
%         opts.compression = 0;
%         opts.maxiters = 200;%maxiters;
%         opts.tol = 1e-6;
%         opts.init = tt_1;
%         %noise_level = sigma_noise^2*numel(data_sel);
% %         noise_level = 1e-2;
%         opts.noise_level = noise_level;% * prod(size(Y));
%
%
%         % ACULRO to data Y
%         tic
%         [tt_2,out2a] = tt_aculro(data_sel,rankR,opts);
%         t2a = toc;


%         err1 = 1-out2a.Fit;
%
%         fig = figure(1); hold on
%         h(2) = plot(err1);
%         set(h(2),'linestyle','--','linewidth',3)
%
%
tt_2x = reshape(full(tt_2),size(tt_2));
tt_2x = bsxfun(@plus,tt_2x,mean_blk);

tt_2x = ipermute(reshape(tt_2x(:,:,:,:,1),blk_size(1),blk_size(2),3,3),[1 3 2 4]);
tt_2x = reshape(tt_2x,blk_size*3);
rec_block = tt_2x;

% imagesc(upd)
end