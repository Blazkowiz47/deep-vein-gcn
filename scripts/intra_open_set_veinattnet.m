function intra_open_set_veinattnet(datasets, seeds, partitionSplit)
    addpath(genpath(pwd));

    if nargin < 1 || isempty(datasets)
        datasets = {"fv300", "fvusm", "mmcbnu"};
    end
    if nargin < 2 || isempty(seeds)
        seeds = 0:4;
    end
    if nargin < 3 || isempty(partitionSplit)
        partitionSplit = 0.8;
    end

    for di = 1:numel(datasets)
        dataset = string(datasets{di});
        for si = 1:numel(seeds)
            statSeed = seeds(si);
            modelName = sprintf("veinAttNet_intra_%s_seed_%d", dataset, statSeed);
            checkpointPath = sprintf("./tmp/%s/checkpoints/best_model.mat", modelName);
            outputDir = sprintf("./features/%s", modelName);

            fprintf("Running intra VeinAttNet export dataset=%s stat_seed=%d\n", dataset, statSeed);
            fprintf("checkpoint=%s\n", checkpointPath);
            fprintf("outputDir=%s\n", outputDir);

            export_left_out_features(checkpointPath, dataset, statSeed, partitionSplit, outputDir);
        end
    end
end


function export_left_out_features(checkpointPath, dataset, statSeed, partitionSplit, outputDir)
    if ~exist(checkpointPath, "file")
        error("Checkpoint not found: %s", checkpointPath);
    end

    datasetRoot = fullfile("./data", dataset, string(statSeed));
    trainDir = fullfile(datasetRoot, "train");
    testDir = fullfile(datasetRoot, "test");
    if ~exist(trainDir, "dir")
        error("Train directory not found: %s", trainDir);
    end
    if ~exist(outputDir, "dir")
        mkdir(outputDir);
    end

    checkpoint = load(checkpointPath, "net");
    net = checkpoint.net;
    featureLayerName = get_feature_layer_name(net);

    leftOutIds = get_left_out_subject_ids(trainDir, partitionSplit);
    for idx = 1:numel(leftOutIds)
        subjectId = leftOutIds{idx};
        subjectOutputDir = fullfile(outputDir, subjectId);
        if ~exist(subjectOutputDir, "dir")
            mkdir(subjectOutputDir);
        end

        process_subject_split(net, featureLayerName, fullfile(trainDir, subjectId), subjectOutputDir, "train");
        process_subject_split(net, featureLayerName, fullfile(testDir, subjectId), subjectOutputDir, "test");
    end
end


function leftOutIds = get_left_out_subject_ids(trainDir, partitionSplit)
    classDirs = dir(trainDir);
    classDirs = classDirs([classDirs.isdir]);
    classDirs = classDirs(~ismember({classDirs.name}, {".", ".."}));
    subjectIds = sort({classDirs.name});
    totalIds = floor(numel(subjectIds) * partitionSplit);
    leftOutIds = subjectIds(totalIds + 1:end);
end


function process_subject_split(net, featureLayerName, subjectInputDir, subjectOutputDir, splitName)
    if ~exist(subjectInputDir, "dir")
        return
    end

    imageFiles = dir(subjectInputDir);
    imageFiles = imageFiles(~[imageFiles.isdir]);
    for imageIndex = 1:numel(imageFiles)
        imageName = imageFiles(imageIndex).name;
        imagePath = fullfile(subjectInputDir, imageName);
        [~, stem, ~] = fileparts(imageName);
        outputTextPath = fullfile(subjectOutputDir, sprintf("%s_%s.txt", splitName, stem));
        if exist(outputTextPath, "file")
            continue
        end

        image = read_and_preprocess_image(imagePath);
        features = activations(net, image, featureLayerName, "OutputAs", "rows");
        features = gather(features);
        features = features(:);
        writematrix(features, outputTextPath);
    end
end


function featureLayerName = get_feature_layer_name(net)
    featureLayer = net.Layers(end - 3);
    featureLayerName = string(featureLayer.Name);
    if strlength(featureLayerName) == 0
        error("Feature layer name is empty. Unable to extract activations.");
    end
end


function image = read_and_preprocess_image(filename)
    image = imread(filename);
    if ismatrix(image)
        image = cat(3, image, image, image);
    end
    image = imresize(image, [224 224]);
end
