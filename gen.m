function gen
addpath ./FCRN-DepthPrediction/matlab/

addpath ./mcg/pre-trained/
install();

% -------------------------------------------------------------------------
% Setup MatConvNet
% -------------------------------------------------------------------------
matconvnet_path = '/export/wangqingze/matconvnet-1.0-beta20';
setupMatConvNet(matconvnet_path);

% -------------------------------------------------------------------------
% Options for depth prediction model
% -------------------------------------------------------------------------
opts.interp = 'nearest';    % interpolation method applied during resizing
opts.imageSize = [460, 345]; % desired image size for evaluation
opts.dataDirImages = '/export/wangqingze/py-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages/';
opts.dataDir = fullfile(pwd, 'depth');     % working directory
netOpts.gpu = false;     % set to true to enable GPU support
netOpts.plot = false;    % set to true to visualize the predictions during inference

net = get_dp_model(opts);

img_list=dir('/export/wangqingze/py-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages/*.jpg');

filename = 'dset.h5';
if exist(filename, 'file')
    delete(filename);
end

for i = 1:numel(img_list)
%for i = 1:1:5
    img = imread(fullfile(opts.dataDirImages, img_list(i).name));  
    h5create(filename, strcat('/image/', img_list(i).name), [size(img, 3), size(img, 2), size(img, 1)], 'Datatype', 'uint8');
    h5write(filename, strcat('/image/', img_list(i).name), permute(img, [3, 2, 1]));
    %imwrite(uint8(img), img_list(i).name, 'jpeg');

    depth = DepthMapPrediction(img, net, netOpts);
    depth = imresize(depth, [size(img, 1), size(img, 2)]);
    h5create(filename, strcat('/depth/', img_list(i).name), [size(depth, 1), size(depth, 2)], 'Datatype', 'single');
    h5write(filename, strcat('/depth/', img_list(i).name), single(depth));

    %seg = im2ucm(img, 'fast');
    seg = im2ucm(img, 'accurate');
    seg = imresize(seg, [size(img, 1), size(img, 2)]);
    level = graythresh(seg);
    seg = im2bw(seg, level);
    seg = ~seg;

    cc = bwconncomp(seg);
    seg = labelmatrix(cc);
    seg = seg + 1;
    % numPixels = cellfun(@numel, cc.PixelIdxList);
    % [biggest, idx] = max(numPixels);
    % seg(cc.PixelIdxList{idx}) = 255;
    % imwrite(seg, img_list(i).name, 'jpeg');

    cnt = tabulate(seg(:));
    area = (cnt(:, 2))';
    label = (cnt(:, 1))';
    h5create(filename, strcat('/seg/', img_list(i).name), [size(depth, 1), size(depth, 2)], 'Datatype', 'uint16');
    h5write(filename, strcat('/seg/', img_list(i).name), uint16(seg));
    h5writeatt(filename, strcat('/seg/', img_list(i).name), 'area', int64(area));
    h5writeatt(filename, strcat('/seg/', img_list(i).name), 'label', int64(label));
end

% -------------------------------------------------------------------------
% get depth prediction model
% -------------------------------------------------------------------------
function net = get_dp_model(opts)
opts.dataDir = fullfile(opts.dataDir, 'models');
if ~exist(opts.dataDir, 'dir'), mkdir(opts.dataDir); end

filename = fullfile(opts.dataDir, 'Make3D_ResNet-UpProj.mat');
if ~exist(filename, 'file')
    url = 'http://campar.in.tum.de/files/rupprecht/depthpred/Make3D_ResNet-UpProj.zip';
    fprintf('downloading trained model: %s\n', url);
    unzip(url, opts.dataDir);
end

net = load(filename);
