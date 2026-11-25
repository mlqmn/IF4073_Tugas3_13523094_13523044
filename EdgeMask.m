function mask = EdgeMask(edges, radius)
mask = edges;
if radius > 0
    [x, y] = meshgrid(-radius:radius);
    disk = double(x.^2 + y.^2 <= radius^2);
    dilated = conv2(double(mask), disk, 'full') > 0;
    mask = conv2(double(dilated), disk, 'valid') == sum(disk(:));
end
[background, ~] = labelRegions(~mask, false);
border = [background(1,:), background(end,:), background(:,1)', background(:,end)'];
mask = mask | (background > 0 & ~ismember(background, border));
[labels, areas] = labelRegions(mask, true);
if isempty(areas), return, end
[~, largest] = max(areas);
mask = labels == largest;
end

function [labels, areas] = labelRegions(mask, diagonal)
[h, w] = size(mask);
labels = zeros(h, w);
areas = zeros(numel(mask), 1);
queue = zeros(numel(mask), 1);
offsets = [-1 0; 1 0; 0 -1; 0 1];
if diagonal, offsets = [offsets; -1 -1; -1 1; 1 -1; 1 1]; end
count = 0;
for seed = find(mask(:))'
    if labels(seed) ~= 0, continue, end
    count = count + 1;
    labels(seed) = count;
    queue(1) = seed;
    head = 1;
    tail = 1;
    while head <= tail
        pixel = queue(head);
        head = head + 1;
        row = mod(pixel - 1, h) + 1;
        col = floor((pixel - 1) / h) + 1;
        for k = 1:size(offsets, 1)
            r = row + offsets(k, 1);
            c = col + offsets(k, 2);
            if r < 1 || r > h || c < 1 || c > w, continue, end
            next = r + (c - 1) * h;
            if mask(next) && labels(next) == 0
                labels(next) = count;
                tail = tail + 1;
                queue(tail) = next;
            end
        end
    end
    areas(count) = tail;
end
areas = areas(1:count);
end
