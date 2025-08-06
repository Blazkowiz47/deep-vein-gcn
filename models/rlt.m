function done=rlt(fnames, onames)
    done = 0;
    for i=1:length(fnames)
        try
            fname =  fnames{i};
            oname =  onames{i};
            img = im2double(load_image(fname));
            fvr = ones(size(img));

            %% Extract veins using repeated line tracking method
            max_iterations = 1000; r=1; W=17; % Parameters

            v_repeated_line_1 = miura_repeated_line_tracking(img,fvr,max_iterations,r,W);
            md = median(v_repeated_line_1(v_repeated_line_1>0)); % Binarise the vein image
            v_repeated_line_bin_1 = v_repeated_line_1 > md;
            features = uint8(v_repeated_line_bin_1);
            save(oname, 'features');

        catch ME
            fprintf('Error processing image %d: %s\n', i, ME.message);
        end
    end
end
