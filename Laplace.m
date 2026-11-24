function [edges] = Laplace(img)
    % Edge detection using laplace operator
    kernel = [
        0,  1,  0;
        1, -4,  1;
        0,  1,  0
    ];
    edges = uint8(convn(double(img), double(kernel)));
end