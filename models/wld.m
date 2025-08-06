function done=wld(fnames, onames)
    done = 0;
    for i=1:length(fnames)
        try
            fname =  fnames{i};
            oname =  onames{i};
            img = load_image(fname);
            fvr = ones(size(img));
            r = 7; t=2; g=41;
            v_wide_line_1 = huang_wide_line(img,fvr,r,t,g);
            features = v_wide_line_1;
            save(oname, 'features');

        catch ME
            fprintf('Error processing image %d: %s\n', i, ME.message);
        end
    end
end
