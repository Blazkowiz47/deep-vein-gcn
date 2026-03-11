function image=load_image(fname)
    image = imread(fname);
    if size(image,1) < size(image,2)
        image = imresize(image,[100,300]);
    else
        image = imresize(image,[300,100]);
    end
end
