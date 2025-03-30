function  veinattnet(traincsv,validationcsv,checkpointPath)
    % Takes in:
    %     traincsv: path to train csv
    %     testcsv: path to test csv
    %     validationcsv: path to validation csv
    %     checkpointPath: path to mat file to save
    %     onnxPath: path to onnx file to save
    
    
    traintb = readtable(traincsv,'TextType', 'string','Delimiter', ',');
    imdsTrain = imageDatastore(traintb.Path, 'Labels', categorical(traintb.Label));
    
    %% 
    % [imdsTrain,~] = splitEachLabel(trainimds,0.9999,"randomized");
    
    classes = categories(categorical(traintb.Label));
    
    pixelRange = [-30 30];
    scaleRange = [0.9 1.1];
    imageAugmenter = imageDataAugmenter( ...
        RandXReflection=true, ...
        RandXTranslation=pixelRange, ...
        RandYTranslation=pixelRange, ...
        RandRotation=[-45 45], ...
        RandXScale=scaleRange, ...
        RandYScale=scaleRange);
    
    inputSize = [224 224 3];
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,DataAugmentation=imageAugmenter, ColorPreprocessing = 'gray2rgb');
    
    %%
    validationtb = readtable(validationcsv,'TextType', 'string','Delimiter', ',');
    imdsValidation = imageDatastore(validationtb.Path, 'Labels', categorical(validationtb.Label));
    
    % [imdsValidation,~] = splitEachLabel(validationimds,0.9999,"randomized");
    imdsValidation.ReadFcn = @readAndPreprocessImage_train_GoogleNet;%@readFunctionTrain;
    augimdsValidation = augmentedImageDatastore([224 224],imdsValidation);
    classes = categories(categorical(validationtb.Label));
    numClasses_V = numel(classes);
    
    
    layers = [
        imageInputLayer([224 224 3])
        convolution2dLayer(7,32,Stride=2,Padding="same")
        groupNormalizationLayer("all-channels")
        reluLayer
        maxPooling2dLayer(3,Stride=2)
    
        convolution2dLayer(5,32,Stride=2,Padding="same")
        groupNormalizationLayer("all-channels")
        reluLayer
        maxPooling2dLayer(3,Stride=2)
    
        convolution2dLayer(3,32,Stride=2,Padding="same")
        groupNormalizationLayer("all-channels")
        reluLayer
        maxPooling2dLayer(3,Stride=2)
    
        globalAveragePooling2dLayer
    
        flattenLayer
        selfAttentionLayer(4,64)
        layerNormalizationLayer
        fullyConnectedLayer(numClasses_V)
        softmaxLayer
        classificationLayer
        ];
    
    
    miniBatchSize = 128;
    numIterationsPerEpoch = floor(augimdsTrain.NumObservations/miniBatchSize);
    
    options = trainingOptions("adam", ...
        MiniBatchSize=miniBatchSize, ...
        Shuffle="every-epoch", ...
        ValidationData=augimdsValidation, ...
        ValidationFrequency=numIterationsPerEpoch, ...
        MaxEpochs=150, ...
        ExecutionEnvironment="gpu",...
        OutputNetwork="best-validation-loss", ...
        Verbose=false);
    display("Starting training...")
    [net,info]= trainNetwork(augimdsTrain,layers,options);
    save(checkpointPath,"net","info");   
end
