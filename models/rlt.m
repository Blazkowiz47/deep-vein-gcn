function score=rlt(img_1,fvr_1, img_2,fvr_2)
    %% Extract veins using repeated line tracking method
    max_iterations = 1000; r=1; W=17; % Parameters

    v_repeated_line_1 = miura_repeated_line_tracking(img_1,fvr_1,max_iterations,r,W);
    md = median(v_repeated_line_1(v_repeated_line_1>0)); % Binarise the vein image
    v_repeated_line_bin_1 = v_repeated_line_1 > md;

    v_repeated_line_2 = miura_repeated_line_tracking(img_2,fvr_2,max_iterations,r,W);
    md = median(v_repeated_line_2(v_repeated_line_2>0)); % Binarise the vein image
    v_repeated_line_bin_2 = v_repeated_line_2 > md;

    score = corr2(uint8(v_repeated_line_bin_1), uint8(v_repeated_line_bin_2));
end
