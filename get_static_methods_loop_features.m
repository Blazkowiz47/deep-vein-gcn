function static_methods_loop
    datasets = ["mmcbnu","fvusm","fv300","polyu","vera"];
    parfor di = 1:length(datasets)
        dataset = datasets(di);
        sprintf("Evaluating trained except %s: on %s 0",dataset,dataset)
        testCsv = sprintf("./data/leaveoutds_%s_seed_0/test.csv",dataset);
        % matlab_test(checkpointPath, testCsv, seed, dataset);
        testtb = readtable(testCsv, "TextType", "string", "Delimiter", ",");
        classes = categories(categorical(testtb.Label));
        cid_map = containers.Map('KeyType', 'char', 'ValueType', 'any');
        for i = 1:height(testtb)
            label = testtb.Label{i};
            path = testtb.Path{i};
            path = replace(path, dataset,sprintf('enhanced_%s',dataset));

            % Check if the label is already a key in the map
            if isKey(cid_map, label)
                tempList = cid_map(label);  % Extract the current list
                tempList{end+1} = path;      % Append the new path
                cid_map(label) = tempList;  % Assign it back
            else
                cid_map(label) = {path};    % Create a new entry
            end
        end

        % Calculate genuine scores
        % For each label, get the list of paths
        % and compare all the images within the same label

        % Save the scores
        methods = {'rlt','mcp'};
        for mi=1:length(methods)
            method=methods{mi};
            if ~exist(sprintf("./tmp/%s/%s",method,dataset), 'dir')
               mkdir(sprintf("./tmp/%s/%s",method,dataset))
            end
            sprintf("Evaluating %s: on %s 0",method,dataset)
            method_map =  containers.Map('KeyType', 'char', 'ValueType', 'any');


            failed_gen = 0;
            files1 = {};
            files2 = {};
            for i = 1:length(classes)
                label = classes{i};
                paths = cid_map(label);
                % Read images and calculate features
                % Compare all pairs of images within the same label
                % Calculate genuine scores
                for j = 1:length(paths)
                    for k = 1:length(paths)
                        if j == k
                            continue;
                        end
                        path1 = paths{j};
                        path2 = paths{k};
                        files1 = [files1; path1];
                        files2 = [files2; path2];
                    end
                end
            end

            scores = zeros(length(files1), 1);
            scores = scores*100;
            sprintf("Calculating %s %s %d",method,dataset,length(files1))

            for i = 1:length(files1)
                path1 = files1{i};
                path2 = files2{i};
                % Perform rlt, wld and mcp comparison here

                try
                    if method == "mcp"
                        score = mcp(path1, path2);
                    elseif method == "wld"
                        score = wld(path1, path2);
                    elseif method == "rlt"
                        score = rlt(path1, path2);
                    end
                    scores(i) = score;
                catch ME
                    % Handle the error here
                    sprintf("Error comparing %s and %s: %s\n", path1, path2, ME.message)
                    continue;  % Skip this iteration
                end
            end

            ogfile =  sprintf("./tmp/%s/%s/genuine.txt",method,dataset);
            writematrix(scores, ogfile);

            files1 = {};
            files2 = {};
            failed_imp=0;
            for i = 1:length(classes)
                for j = 1:length(classes)
                    label1 = classes{i};
                    label2 = classes{j};
                    paths1 = cid_map(label1);
                    paths2 = cid_map(label2);
                    % Read images and calculate features
                    % Compare all pairs of images within the same label
                    % Calculate genuine scores
                    for k = 1:length(paths1)
                        for l = 1:length(paths2)
                            path1 = paths1{k};
                            path2 = paths2{l};
                            % Perform rlt, wld and mcp comparison here
                            files1 = [files1; path1];
                            files2 = [files2; path2];
                        end
                    end
                end
            end

            sprintf("Calculating %s %s %d",method,dataset,length(files1))
            scores = zeros(length(files1), 1);
            for i = 1:length(files1)
                path1 = files1{i};
                path2 = files2{i};
                % Perform rlt, wld and mcp comparison here

                try
                    if method == "mcp"
                        score = mcp(path1, path2);
                    elseif method == "wld"
                        score = wld(path1, path2);
                    elseif method == "rlt"
                        score = rlt(path1, path2);
                    end
                    scores(i) = score;
                catch ME
                    % Handle the error here
                    sprintf("Error comparing %s and %s: %s\n", path1, path2, ME.message)
                    continue;  % Skip this iteration
                end
            end
            oifile =  sprintf("./tmp/%s/%s/imposter.txt",method,dataset);
            writematrix(scores, oifile);
            sprintf("Failed %s in %s: %d genuine, %d imposter",method,dataset,failed_gen,failed_imp)
        end

    end
end

