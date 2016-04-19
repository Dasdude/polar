function [net, info] = cnn_cifar_polar(varargin)
% CNN_CIFAR   Demonstrates MatConvNet on CIFAR-10
%    The demo includes two standard model: LeNet and Network in
%    Network (NIN). Use the 'modelType' option to choose one.

% run(fullfile(fileparts(mfilename('fullpath')), ...
%     '..', 'matlab', 'vl_setupnn.m')) ;

opts.modelType = 'lenet' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

switch opts.modelType
    case 'lenet'
        
        %opts.train.learningRate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,5)] ;
        opts.train.learningRate = [0.05*ones(1,100) 0.005*ones(1,10) 0.0005*ones(1,5)] ;
        opts.train.weightDecay = 0.0001 ;
    case 'nin'
        opts.train.learningRate = [0.5*ones(1,30) 0.1*ones(1,10) 0.02*ones(1,10)] ;
        opts.train.weightDecay = 0.0005 ;
    otherwise
        error('Unknown model type %s', opts.modelType) ;
end
opts.expDir = fullfile('data', sprintf('cifar-%s', opts.modelType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data','cifar') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.whitenData = false ;
opts.contrastNormalization = false ;
opts.train.batchSize = 750  ;
opts.train.continue = false;
opts.train.gpus = 1 ;
opts.train.expDir = opts.expDir ;
% uses polar coordinates
opts.usePolar =true;
opts.useGmm = false;
opts.type = 2; % 0 is log 1 is lin 2 is square
% opts.upSampleRate = double(2);
% opts.filterSigma = single(2/3);
% opts.interval = 6;
% opts.kernel = single(fspecial('gaussian',ceil(double(opts.filterSigma *3)),double(opts.filterSigma)));
% opts.extrapval = single(0);
opts.plotDiagnostics = false;
opts.prefetch = false;

if opts.usePolar
opts = updateOptsPolar(opts,0,2,2/3,6,0);
end
opts = vl_argparse(opts, varargin) ;


% ---------------- ----------------------------------------------------
%                                               Prepare data and model
% --------------------------------------------------------------------

switch opts.modelType
    case 'lenet', net = cnn_cifar_init_polar(opts,opts) ;
    case 'nin',   net = cnn_cifar_init_nin(opts,opts) ;
end

if exist(opts.imdbPath, 'file')
    
    imdb = load(opts.imdbPath) ;
else
    imdb = getCifarImdb(opts) ;
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
end
%imdb.images.data = pol_transform(imdb.images.data)
%Im_Struct = load('cifar_polar.mat');
%imdb.images.data = Im_Struct.output_args;
% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
if opts.useGmm
    imdb = createCentHist(imdb,3);
else
    imdb = createCentHist(imdb,1);
end

[net, info,imdb] = cnn_train_polar(net, imdb, @getBatch, ...
    opts.train, ...
    'val', find(imdb.images.set == 3),'usePolar',opts.usePolar,'plotDiagnostics',opts.plotDiagnostics,'prefetch',opts.prefetch,'useGmm',opts.useGmm) ;

% --------------------------------------------------------------------
function [im, labels, centHist,imdb,isFliped] = getBatch(imdb, batch,net,isPolar,useGmm)
% --------------------------------------------------------------------
isFliped = false;
im = imdb.images.data(:,:,:,batch) ;
im = gpuArray(im);
centHist = [];
if useGmm
    imdb = updateSigmas(imdb,batch,net);
end

if isPolar
    
    if rand > 0.5 && ~useGmm,
        
        
        im=fliplr(im) ;
        [centHist,imdb,batch] = getCentersImdb(imdb,batch,true);
        isFliped = true;
    else
        [centHist,imdb,batch] = getCentersImdb(imdb,batch,false);
        isFliped = false;
    end
    
end

labels = imdb.images.labels(1,batch) ;


% --------------------------------------------------------------------
function imdb = getCifarImdb(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
    {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
    fprintf('downloading %s\n', url) ;
    untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
    fd = load(files{fi}) ;
    data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
    labels{fi} = fd.labels' + 1; % Index from 1
    sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));

% remove mean in any case
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

if opts.contrastNormalization
    z = reshape(data,[],60000) ;
    z = bsxfun(@minus, z, mean(z,1)) ;
    n = std(z,0,1) ;
    z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
    data = reshape(z, 32, 32, 3, []) ;
    save('./data/cifar/cifar_stats/data_cifar_std.mat','n');
    save('./data/cifar/cifar_stats/data_cifar_mean.mat','dataMean');
end

if opts.whitenData
    z = reshape(data,[],60000) ;
    W = z(:,set == 1)*z(:,set == 1)'/60000 ;
    [V,D] = eig(W) ;
    % the scale is selected to approximately preserve the norm of W
    d2 = diag(D) ;
    en = sqrt(mean(d2)) ;
    z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
    data = reshape(z, 32, 32, 3, []) ;
    save('./data/cifar/cifar_stats/data_cifar_eigval_sqmean.mat','en');
    save('./data/cifar/cifar_stats/data_cifar_eigval.mat','d2');
    save('./data/cifar/cifar_stats/data_cifar_eigvec.mat','V');
end

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;
