function score=mcp(img_1,fvr_1, img_2,fvr_2)
    sigma = 3; % Parameter
    v_max_curvature_1 = miura_max_curvature(img_1,fvr_1,sigma);
    md = median(v_max_curvature_1(v_max_curvature_1>0)); % Binarise the vein image
    v_max_curvature_bin_1 = v_max_curvature_1 > md;

    v_max_curvature_2 = miura_max_curvature(img_2,fvr_2,sigma);
    md = median(v_max_curvature_2(v_max_curvature_2>0)); % Binarise the vein image
    v_max_curvature_bin_2 = v_max_curvature_2 > md;

    score = corr2(uint8(v_max_curvature_bin_1), uint8(v_max_curvature_bin_2));
end
