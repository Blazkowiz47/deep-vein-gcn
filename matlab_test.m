function matlab_test (checkpointPath, testCsv, seed, leavoutds)

    load(checkpointPath, "net", "info");
    disp(info);
    loopsize = size(info.ValidationAccuracy);
    disp(info.ValidationAccuracy(loopsize(2)-10:loopsize(2)));
    testtb = readtable(testCsv, "TextType", "string", "Delimiter", ",");
    paths = testtb.Path;
    outputPaths = testtb.OPath;
    % labels = categorical(testtb.Label);
    for i = 1:length(paths)
        img = readAndPreprocessImage_train_GoogleNet(paths(i));
        img = imresize(img, [224 224]);
        feature = activations(net,img,"layernorm");
        outputPath = outputPaths(i);
        outputPath = replace(outputPath, 'features',sprintf('features/leaveout_%s/%s',leavoutds,seed));
        dirPath = fileparts(outputPath);
        if ~exist(dirPath, 'dir')
            mkdir(dirPath);
        end

        writematrix(feature, outputPath);
    end







