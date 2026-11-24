function [labels, segmented] = SplitAndMerge(img, splitThreshold, minSize, mergeThreshold)
if nargin < 2, splitThreshold = 18; end
if nargin < 3, minSize = 8; end
if nargin < 4, mergeThreshold = 22; end

img = double(img);
if ndims(img) == 2, img = reshape(img, size(img,1), size(img,2), 1); end
[h, w, channels] = size(img);
labels = zeros(h, w);
nextLabel = 1;
split(1, h, 1, w);
leafCount = nextLabel - 1;

pixelLabels = labels(:);
pixels = reshape(img, [], channels);
counts = accumarray(pixelLabels, 1, [leafCount 1]);
sums = zeros(leafCount, channels);
for k = 1:channels
    sums(:,k) = accumarray(pixelLabels, pixels(:,k), [leafCount 1]);
end

left = labels(:,1:end-1); right = labels(:,2:end);
top = labels(1:end-1,:); bottom = labels(2:end,:);
pairs = unique([left(:), right(:); top(:), bottom(:)], 'rows');
pairs(pairs(:,1) == pairs(:,2), :) = [];

parent = 1:leafCount;
changed = true;
while changed
    changed = false;
    for p = 1:size(pairs,1)
        a = root(pairs(p,1));
        b = root(pairs(p,2));
        if a ~= b && norm(sums(a,:)/counts(a) - sums(b,:)/counts(b)) <= mergeThreshold
            parent(b) = a;
            counts(a) = counts(a) + counts(b);
            sums(a,:) = sums(a,:) + sums(b,:);
            changed = true;
        end
    end
end

for id = 1:leafCount, parent(id) = root(id); end
labels = reshape(parent(labels(:)), h, w);
[~, ~, labels] = unique(labels);
labels = reshape(labels, h, w);

pixelLabels = labels(:);
regionCount = max(pixelLabels);
means = zeros(regionCount, channels);
for k = 1:channels
    means(:,k) = accumarray(pixelLabels, pixels(:,k), [regionCount 1], @mean);
end
segmented = reshape(means(pixelLabels,:), h, w, channels);
if channels == 1, segmented = segmented(:,:,1); end
segmented = uint8(round(segmented));

    function split(r1, r2, c1, c2)
        region = reshape(img(r1:r2, c1:c2, :), [], channels);
        if max(std(region, 0, 1)) <= splitThreshold || r2-r1+1 <= minSize || c2-c1+1 <= minSize
            labels(r1:r2, c1:c2) = nextLabel;
            nextLabel = nextLabel + 1;
            return
        end
        rm = floor((r1+r2)/2);
        cm = floor((c1+c2)/2);
        split(r1, rm, c1, cm);
        split(r1, rm, cm+1, c2);
        split(rm+1, r2, c1, cm);
        split(rm+1, r2, cm+1, c2);
    end

    function id = root(id)
        while parent(id) ~= id, id = parent(id); end
    end
end
