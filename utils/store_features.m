function store_features(checkpointPath, dataDir, outputDir)
    if ~exist(checkpointPath, 'file')
        error("Checkpoint not found: %s", checkpointPath);
    end

    if ~exist(dataDir, 'dir')
        error("Data directory not found: %s", dataDir);
    end

    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    logPath = fullfile(outputDir, "store_features.log");
    write_log(logPath, "RUN START checkpoint=%s dataDir=%s outputDir=%s", checkpointPath, dataDir, outputDir);

    checkpoint = load(checkpointPath, "net");
    net = checkpoint.net;
    featureLayerName = get_feature_layer_name(net);
    write_log(logPath, "CHECKPOINT LOADED featureLayer=%s", featureLayerName);

    splits = {"train", "test"};
    for splitIndex = 1:numel(splits)
        splitName = splits{splitIndex};
        splitInputDir = fullfile(dataDir, splitName);
        if ~exist(splitInputDir, 'dir')
            write_log(logPath, "SPLIT MISSING split=%s path=%s", splitName, splitInputDir);
            continue
        end

        splitOutputDir = fullfile(outputDir, splitName);
        if ~exist(splitOutputDir, 'dir')
            mkdir(splitOutputDir);
        end
        write_log(logPath, "SPLIT START split=%s input=%s output=%s", splitName, splitInputDir, splitOutputDir);

        classDirs = dir(splitInputDir);
        classDirs = classDirs([classDirs.isdir]);
        classDirs = classDirs(~ismember({classDirs.name}, {'.', '..'}));

        tasks = struct('imagePath', {}, 'outputPath', {}, 'outputTextPath', {});
        for classIndex = 1:numel(classDirs)
            className = classDirs(classIndex).name;
            classInputDir = fullfile(splitInputDir, className);
            classOutputDir = fullfile(splitOutputDir, className);
            if ~exist(classOutputDir, 'dir')
                mkdir(classOutputDir);
            end

            imageFiles = dir(classInputDir);
            imageFiles = imageFiles(~[imageFiles.isdir]);

            for imageIndex = 1:numel(imageFiles)
                imageName = imageFiles(imageIndex).name;
                imagePath = fullfile(classInputDir, imageName);
                [~, stem, ~] = fileparts(imageName);
                outputPath = fullfile(classOutputDir, [stem '.mat']);
                outputTextPath = fullfile(classOutputDir, [stem '.txt']);

                if exist(outputTextPath, 'file')
                    continue
                end

                tasks(end + 1) = struct( ...
                    'imagePath', imagePath, ...
                    'outputPath', outputPath, ...
                    'outputTextPath', outputTextPath);
            end
        end

        taskCountMessage = sprintf("TASKS READY split=%s count=%d", splitName, numel(tasks));
        disp(taskCountMessage);
        write_log(logPath, "%s", taskCountMessage);

        parfor taskIndex = 1:numel(tasks)
            imagePath = tasks(taskIndex).imagePath;
            outputPath = tasks(taskIndex).outputPath;
            outputTextPath = tasks(taskIndex).outputTextPath;

            try
                image = read_and_preprocess_image(imagePath);
            catch
                continue
            end

            features = activations(net, image, featureLayerName, "OutputAs", "rows");
            features = gather(features);
            features = features(:);
            save(outputPath, "features");
            writematrix(features, outputTextPath);
        end
    end

    write_log(logPath, "RUN END checkpoint=%s", checkpointPath);
end

function featureLayerName = get_feature_layer_name(net)
    featureLayer = net.Layers(end-3);
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

function write_log(logPath, message, varargin)
    fid = fopen(logPath, 'a');
    if fid == -1
        error("Unable to open log file: %s", logPath);
    end

    timestamp = char(datetime("now", "Format", "yyyy-MM-dd HH:mm:ss"));
    formatMessage = char(message);
    fprintf(fid, "[%s] ", timestamp);
    fprintf(fid, [formatMessage '\n'], varargin{:});
    fclose(fid);
end
