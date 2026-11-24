function [edges] = Roberts(img)
    kernel_p = [
        1,   0;
        0,  -1;
        ];
    kernel_m = [
         0,  1;
        -1,  0;
        ];
    edges = uint8(abs(convn(double(img), double(kernel_p))) + ...
            abs(convn(double(img), double(kernel_m))));
end