function [edges, segmented] = EdgeSegment(img, method, threshold, kernelSize, sigma, radius)
gray = img;
if size(img, 3) == 3
    gray = uint8(0.299 * double(img(:,:,1)) + ...
        0.587 * double(img(:,:,2)) + 0.114 * double(img(:,:,3)));
end
if strcmp(method, 'Canny')
    if exist('edge', 'file') == 0
        error('Canny memerlukan Image Processing Toolbox. Pilih operator lain.');
    end
    edges = edge(gray, 'Canny', threshold, sigma);
else
    if strcmp(method, 'LoG')
        if mod(kernelSize, 2) == 0, error('Ukuran kernel LoG harus ganjil.'); end
        response = LoG(gray, kernelSize, sigma);
    else
        response = feval(method, gray);
    end
    [h, w] = size(gray);
    row = floor((size(response, 1) - h) / 2);
    col = floor((size(response, 2) - w) / 2);
    response = response(row + (1:h), col + (1:w));
    edges = double(response) / 255 >= threshold;
end
mask = EdgeMask(edges, radius);
if ~any(mask(:))
    error('Tidak ada region terdeteksi. Turunkan threshold atau ubah radius closing.');
end
segmented = img .* uint8(mask);
end
