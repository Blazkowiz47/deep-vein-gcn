function done=mcp(fnames, onames)
    done = 0;
    for i=1:length(fnames)
        try
            fname =  fnames{i};
            oname =  onames{i};
            img = im2double(load_image(fname));
            fvr = ones(size(img));

            sigma = 3; % Parameter
            v_max_curvature_1 = miura_max_curvature(img,fvr,sigma);
            md = median(v_max_curvature_1(v_max_curvature_1>0)); % Binarise the vein image
            v_max_curvature_bin_1 = v_max_curvature_1 > md;
            features = uint8(v_max_curvature_bin_1);
            save(oname, 'features');

        catch ME
            fprintf('Error processing image %d: %s\n', i, ME.message);
        end
    end
end
