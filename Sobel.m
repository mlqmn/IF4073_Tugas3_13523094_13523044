function [edges] = Sobel(img)
    kernel_y = [
        1,   2,  1;
        0,   0,  0;
        -1, -2, -1;
        ];
    kernel_x = [
        1, 0, -1;
        2, 0, -2;
        1, 0, -1;
        ];
    edges = uint8(sqrt( ...
        convn(double(img), double(kernel_y)).^2 + ...
        convn(double(img), double(kernel_x)).^2 ...
        ));

end