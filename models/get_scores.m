function scores=get_scores(paths1,paths2)
    scores = zeros(length(paths1),1);

    parfor i = 1:length(paths1)
        try
            % Load the features
            featf1 = load(paths1{i});
            feat1 = featf1.features;
            featf2 = load(paths2{i});
            feat2 = featf2.features;
            score =  corr2(feat1,feat2);
            % display(score);
            scores(i) =score;
        catch ME
            fprintf('Error processing image pair %d: %s\n', i, ME.message);
            % msgText = getReport(ME, 'extended');
            % fprintf(2, '%s\n', msgText); % Print to standard error
        end
    end

end
