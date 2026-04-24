function benchmark_veinattnet_runtime()
    addpath(genpath(pwd));

    dataset = "fv300";
    seed = "0";
    warmupIters = 20;
    timedIters = 200;
    batchSize = 1;
    outputPath = "ablation/fv300_veinattnet_runtime.json";

    checkpointPath = sprintf("./final_runs/leaveoutds_veinAttNet_%s_seed_%s/best_model.mat", dataset, seed);
    if ~exist(checkpointPath, "file")
        error("Checkpoint not found: %s", checkpointPath);
    end

    checkpoint = load(checkpointPath, "net");
    net = checkpoint.net;
    featureLayerName = get_feature_layer_name(net);

    dummyInput = make_dummy_input(batchSize);

    cpuMetrics = benchmark_environment(net, dummyInput, featureLayerName, "cpu", warmupIters, timedIters);
    gpuMetrics = struct("latency_ms", NaN, "throughput_samples_per_s", NaN);
    if can_use_gpu()
        gpuMetrics = benchmark_environment(net, dummyInput, featureLayerName, "gpu", warmupIters, timedIters);
    end

    results = struct();
    results.method = "VeinAttNet";
    results.dataset = dataset;
    results.seed = str2double(seed);
    results.batch_size = batchSize;
    results.feature_layer = featureLayerName;
    results.params = count_learnable_params(net);
    results.cpu = cpuMetrics;
    results.gpu = gpuMetrics;

    outputDir = fileparts(outputPath);
    if strlength(outputDir) > 0 && ~exist(outputDir, "dir")
        mkdir(outputDir);
    end

    jsonText = jsonencode(results, PrettyPrint=true);
    fid = fopen(outputPath, "w");
    if fid == -1
        error("Could not open output file: %s", outputPath);
    end
    fwrite(fid, jsonText, "char");
    fclose(fid);

    fprintf("| Method | Params (M) | CPU Latency (ms) | CPU Throughput | GPU Latency (ms) | GPU Throughput |\n");
    fprintf("|---|---|---|---|---|---|\n");
    fprintf("| VeinAttNet | %.2f | %.2f | %.2f | %.2f | %.2f |\n", ...
        results.params / 1e6, ...
        results.cpu.latency_ms, ...
        results.cpu.throughput_samples_per_s, ...
        results.gpu.latency_ms, ...
        results.gpu.throughput_samples_per_s);
    fprintf("\nSaved JSON to %s\n", outputPath);
end


function metrics = benchmark_environment(net, dummyInput, featureLayerName, executionEnvironment, warmupIters, timedIters)
    run_forward(net, dummyInput, featureLayerName, executionEnvironment);
    if executionEnvironment == "gpu"
        wait(gpuDevice);
    end

    for idx = 1:warmupIters
        run_forward(net, dummyInput, featureLayerName, executionEnvironment);
    end
    if executionEnvironment == "gpu"
        wait(gpuDevice);
    end

    startTime = tic;
    for idx = 1:timedIters
        run_forward(net, dummyInput, featureLayerName, executionEnvironment);
    end
    if executionEnvironment == "gpu"
        wait(gpuDevice);
    end
    elapsed = toc(startTime);

    metrics = struct();
    metrics.latency_ms = (elapsed / timedIters) * 1000.0;
    metrics.throughput_samples_per_s = (timedIters * size(dummyInput, 4)) / elapsed;
end


function run_forward(net, dummyInput, featureLayerName, executionEnvironment)
    activations( ...
        net, ...
        dummyInput, ...
        featureLayerName, ...
        "OutputAs", "rows", ...
        "ExecutionEnvironment", executionEnvironment);
end


function dummyInput = make_dummy_input(batchSize)
    dummyInput = rand([224, 224, 3, batchSize], "single");
end


function featureLayerName = get_feature_layer_name(net)
    featureLayer = net.Layers(end - 3);
    featureLayerName = string(featureLayer.Name);

    if strlength(featureLayerName) == 0
        error("Feature layer name is empty. Unable to extract activations.");
    end
end


function tf = can_use_gpu()
    tf = false;
    try
        tf = canUseGPU;
        if tf
            gpuDevice;
        end
    catch
        tf = false;
    end
end


function totalParams = count_learnable_params(net)
    totalParams = 0;
    learnableFields = { ...
        "Weights", ...
        "Bias", ...
        "Scale", ...
        "Offset", ...
        "InputWeights", ...
        "RecurrentWeights" ...
    };

    for idx = 1:numel(net.Layers)
        layer = net.Layers(idx);
        for fieldIdx = 1:numel(learnableFields)
            fieldName = learnableFields{fieldIdx};
            if isprop(layer, fieldName)
                value = layer.(fieldName);
                if isnumeric(value) || islogical(value)
                    totalParams = totalParams + numel(value);
                end
            end
        end
    end
end
