function [edges] = LoG(img, kernel_size, sigma)
    k = floor(kernel_size/2);
    [x, y] = meshgrid(-k : k, -k: k);
    r = x.^2 + y.^2;
    kernel = ((r - 2 * sigma ^ 2)/sigma^4) .* exp(-r/(2*sigma^2));
    kernel = kernel - mean(kernel(:));
    edges = uint8(convn(double(img), double(kernel)));
end