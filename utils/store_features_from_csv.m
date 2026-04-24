function store_features_from_csv(taskCsvPath)
    if ~exist(taskCsvPath, 'file')
        error("Task CSV not found: %s", taskCsvPath);
    end

    totalTimer = tic;
    fprintf("READ TASK CSV START path=%s\n", taskCsvPath);
    tasks = readtable(taskCsvPath, ...
        "FileType", "text", ...
        "Delimiter", ",", ...
        "ReadVariableNames", true, ...
        "TextType", "string", ...
        "VariableNamingRule", "preserve");
    taskCount = height(tasks);
    fprintf("TASK CSV LOADED path=%s count=%d elapsed=%.2fs\n", taskCsvPath, taskCount, toc(totalTimer));
    fprintf("TASK CSV COLUMNS %s\n", strjoin(string(tasks.Properties.VariableNames), ", "));

    if taskCount == 0
        fprintf("NO TASKS TO RUN\n");
        return
    end

    requiredColumns = ["checkpointPath", "imagePath", "outputPath", "outputTextPath"];
    missingColumns = setdiff(requiredColumns, string(tasks.Properties.VariableNames));
    if ~isempty(missingColumns)
        error("Task CSV is missing columns: %s", strjoin(missingColumns, ", "));
    end

    fprintf("OUTPUT DIR CREATE SKIPPED expected pre-created by task CSV generator\n");

    checkpointPaths = unique(tasks.checkpointPath, "stable");
    fprintf("CHECKPOINT GROUPS READY count=%d\n", numel(checkpointPaths));
    pool = gcp('nocreate');
    if isempty(pool)
        fprintf("PARALLEL POOL STATUS none active; MATLAB may start one at first parfor\n");
    else
        fprintf("PARALLEL POOL STATUS active workers=%d\n", pool.NumWorkers);
    end

    for checkpointIndex = 1:numel(checkpointPaths)
        checkpointTimer = tic;
        checkpointPath = checkpointPaths(checkpointIndex);
        checkpointTaskMask = tasks.checkpointPath == checkpointPath;
        checkpointTasks = tasks(checkpointTaskMask, :);

        fprintf("CHECKPOINT START %d/%d path=%s tasks=%d\n", ...
            checkpointIndex, numel(checkpointPaths), checkpointPath, height(checkpointTasks));

        loadTimer = tic;
        fprintf("CHECKPOINT LOAD START path=%s\n", checkpointPath);
        checkpoint = load(checkpointPath, "net");
        net = checkpoint.net;
        featureLayerName = get_feature_layer_name(net);
        fprintf("CHECKPOINT LOAD DONE path=%s featureLayer=%s elapsed=%.2fs\n", ...
            checkpointPath, featureLayerName, toc(loadTimer));
        imagePaths = checkpointTasks.imagePath;
        outputPaths = checkpointTasks.outputPath;
        outputTextPaths = checkpointTasks.outputTextPath;
        checkpointTaskCount = numel(imagePaths);

        fprintf("PARFOR START checkpoint=%d/%d tasks=%d\n", ...
            checkpointIndex, numel(checkpointPaths), checkpointTaskCount);
        parfor taskIndex = 1:checkpointTaskCount
            imagePath = imagePaths(taskIndex);
            outputPath = outputPaths(taskIndex);
            outputTextPath = outputTextPaths(taskIndex);

            try
                image = read_and_preprocess_image(imagePath);
            catch
                continue
            end

            features = activations(net, image, featureLayerName, "OutputAs", "rows");
            features = gather(features);
            features = features(:);
            save_feature_outputs(outputPath, outputTextPath, features);
        end

        fprintf("PARFOR DONE checkpoint=%d/%d tasks=%d elapsed=%.2fs\n", ...
            checkpointIndex, numel(checkpointPaths), checkpointTaskCount, toc(checkpointTimer));
        fprintf("CHECKPOINT DONE path=%s\n", checkpointPath);
    end

    fprintf("ALL TASKS DONE count=%d elapsed=%.2fs\n", taskCount, toc(totalTimer));
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

function save_feature_outputs(outputPath, outputTextPath, features)
    save(outputPath, "features");
    writematrix(features, outputTextPath);
end
