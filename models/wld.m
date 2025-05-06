function score=wld(fname_1, fname_2)
    img_1 = imread(fname_1);
    img_2 = imread(fname_2);

    %% Extract veins using repeated line tracking method
    r = 6; t=1; g=21;
    fvr_1 = lee_region(img_1,4,40);    % Get finger region
    v_wide_line_1 = huang_wide_line(img_1,fvr_1,r,t,g);
    md = median(v_wide_line_1(v_wide_line_1>0)); % Binarise the vein image
    v_wide_line_bin_1 = v_wide_line_1 > md;

    fvr_2 = lee_region(img_2,4,40);    % Get finger region
    v_wide_line_2 = huang_wide_line(img_2,fvr_2,r,t,g);
    md = median(v_wide_line_2(v_wide_line_2>0)); % Binarise the vein image
    v_wide_line_bin_2 = v_wide_line_2 > md;

    score = corr2(uint8(v_wide_line_bin_1), uint8(v_wide_line_bin_2));
end
