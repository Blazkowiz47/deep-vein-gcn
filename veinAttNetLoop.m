function veinAttNetLoop
    addpath(genpath(pwd));
    datasets = ["vera","mmcbnu","fvusm","fv300","polyu"];
    seeds = ["0","1","2","3"];
    for si = 1:length(seeds)
        seed = seeds(si);
        for di = 1:length(datasets)
            datasets = ["vera","mmcbnu","fvusm","fv300","polyu"];
            dataset = datasets(di);

            % sprintf("Training except %s: with seed %s ",dataset, seed)
            % trainCsv = sprintf("./data/leaveoutds_%s_seed_%s/train.csv",dataset, seed);
            % validationCsv = sprintf("./data/leaveoutds_%s_seed_%s/validation.csv",dataset, seed);
            % if ~exist(trainCsv, 'file') || ~exist(validationCsv, 'file')
            %     disp("Dataset not found")
            % end

            checkpointPath = sprintf("./tmp/leaveoutds_veinAttNet_%s_seed_%s/checkpoints/best_model.mat",dataset, seed);
            % saveDir = sprintf("./tmp/leaveoutds_veinAttNet_%s_seed_%s_150/checkpoints",dataset, seed);
            % if ~exist(saveDir, 'dir')
            %     mkdir(saveDir);
            % end
            % if ~exist(checkpointPath, 'file')
            %     veinattnet(trainCsv,validationCsv,checkpointPath);
            % end
            dataDir = sprintf("./data/%s/0", dataset);
            outputDir = sprintf("./features/leaveoutds_veinAttNet_%s_seed_%s", dataset, seed);
            fprintf("Evaluating trained except %s: on %s %s\n", dataset, dataset, seed);
            fprintf("checkpointPath=%s\n", checkpointPath);
            fprintf("outputDir=%s\n", outputDir);
            store_features(checkpointPath, dataDir, outputDir);
        end
    end
end
