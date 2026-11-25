function fig = SegmentasiGUI
folder = fileparts(mfilename('fullpath'));
img = [];
imagePath = '';

fig = uifigure('Name', 'Segmentasi Citra', 'Position', [100 100 1200 650]);
layout = uigridlayout(fig, [3 1]);
layout.RowHeight = {32, '1x', 24};
toolbar = uigridlayout(layout, [1 2]);
toolbar.ColumnWidth = {120, '1x'};
toolbar.Padding = [0 0 0 0];
uibutton(toolbar, 'Text', 'Buka citra', 'ButtonPushedFcn', @openImage);
fileLabel = uilabel(toolbar, 'Text', 'Belum ada citra');
tabs = uitabgroup(layout);
status = uilabel(layout, 'Text', 'Pilih citra, atur parameter, lalu klik Proses.');

[controls, edgeAxes] = makeTab(tabs, 'B. Deteksi Tepi', ...
    {'Citra asli','Deteksi tepi','Segmentasi objek'}, 5, @(~,~) process(1));
uilabel(controls, 'Text', 'Operator');
uilabel(controls, 'Text', 'Threshold (0-1)');
uilabel(controls, 'Text', 'Ukuran kernel LoG');
uilabel(controls, 'Text', 'Sigma LoG / Canny');
uilabel(controls, 'Text', 'Radius closing');
operator = uidropdown(controls, 'Items', ...
    {'Laplace','LoG','Sobel','Prewitt','Roberts','Canny'}, 'Value', 'Sobel');
edgeThreshold = uieditfield(controls, 'numeric', 'Value', 0.2, 'Limits', [0.001 1]);
kernelSize = uieditfield(controls, 'numeric', 'Value', 5, ...
    'Limits', [3 51], 'RoundFractionalValues', 'on');
sigma = uieditfield(controls, 'numeric', 'Value', 1.4, 'Limits', [0.1 20]);
radius = uieditfield(controls, 'numeric', 'Value', 3, ...
    'Limits', [0 50], 'RoundFractionalValues', 'on');

[controls, splitAxes] = makeTab(tabs, 'C. Split and Merge', ...
    {'Citra asli','Hasil split and merge'}, 3, @(~,~) process(2));
uilabel(controls, 'Text', 'Split threshold');
uilabel(controls, 'Text', 'Ukuran minimum');
uilabel(controls, 'Text', 'Merge threshold');
splitThreshold = uieditfield(controls, 'numeric', 'Value', 18, 'Limits', [0 255]);
minSize = uieditfield(controls, 'numeric', 'Value', 8, ...
    'Limits', [1 Inf], 'RoundFractionalValues', 'on');
mergeThreshold = uieditfield(controls, 'numeric', 'Value', 22, 'Limits', [0 442]);

[classNames, classColors] = CamVidClasses;
[controls, semanticAxes, semanticTab] = makeTab(tabs, 'D. Segmentasi Semantik', ...
    {'Citra masukan','Segmentasi semantik','Overlay','Ground truth'}, ...
    numel(classNames) + 1, @(~,~) process(3));
opacityLabel = uilabel(controls, 'Text', 'Opacity overlay');
opacityLabel.Layout.Row = 1;
opacityLabel.Layout.Column = 1;
legendLabel = uilabel(controls, 'Text', 'Kelas CamVid');
legendLabel.Layout.Row = 1;
legendLabel.Layout.Column = [2 numel(classNames) + 1];
overlayOpacity = uieditfield(controls, 'numeric', 'Value', 0.5, 'Limits', [0 1]);
overlayOpacity.Layout.Row = 2;
overlayOpacity.Layout.Column = 1;
for k = 1:numel(classNames)
    swatch = uilabel(controls, 'Text', classNames{k}, 'HorizontalAlignment', 'center', ...
        'BackgroundColor', classColors(k,:) / 255, 'FontColor', readableTextColor(classColors(k,:)));
    swatch.Layout.Row = 2;
    swatch.Layout.Column = k + 1;
end

    function openImage(~, ~)
        startFolder = fullfile(folder, 'bsds500_test_8', '*.jpg');
        if tabs.SelectedTab == semanticTab
            startFolder = fullfile(folder, 'data', 'camvid', 'test', 'images', '*.png');
        end
        [file, path] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff', 'Citra'}, ...
            'Buka citra', startFolder);
        if isequal(file, 0), return, end
        try
            [newImage, map] = imread(fullfile(path, file));
            if ~isempty(map) || ~isa(newImage, 'uint8') || ~ismember(size(newImage, 3), [1 3])
                error('Gunakan citra uint8 grayscale atau RGB, bukan citra indexed.');
            end
            img = newImage;
            imagePath = fullfile(path, file);
            fileLabel.Text = file;
            for ax = [edgeAxes, splitAxes, semanticAxes]
                cla(ax);
            end
            for ax = [edgeAxes(1), splitAxes(1)]
                showImage(ax, img, 'Citra asli');
            end
            showImage(semanticAxes(1), img, 'Citra masukan');
            status.Text = 'Citra dimuat. Pilih tab dan klik Proses.';
        catch err
            uialert(fig, err.message, 'Gagal membuka citra');
        end
    end

    function process(method)
        if isempty(img)
            uialert(fig, 'Buka citra terlebih dahulu.', 'Citra belum dipilih');
            return
        end
        status.Text = 'Memproses citra...';
        drawnow;
        started = tic;
        detail = '';
        try
            switch method
                case 1
                    [edges, result] = EdgeSegment(img, operator.Value, ...
                        edgeThreshold.Value, kernelSize.Value, sigma.Value, radius.Value);
                    showImage(edgeAxes(2), edges, ['Tepi ' operator.Value]);
                    showImage(edgeAxes(3), result, 'Segmentasi objek');
                case 2
                    [labels, result] = SplitAndMerge(img, splitThreshold.Value, ...
                        minSize.Value, mergeThreshold.Value);
                    showImage(splitAxes(2), result, sprintf('Hasil (%d region)', max(labels(:))));
                case 3
                    labels = SemanticSegment(imagePath);
                    colored = colorizeLabels(labels, classColors);
                    opacity = overlayOpacity.Value;
                    overlay = uint8((1 - opacity) * double(img) + opacity * double(colored));
                    showImage(semanticAxes(2), colored, 'Segmentasi semantik');
                    showImage(semanticAxes(3), overlay, 'Overlay');
                    detail = showGroundTruth(labels);
            end
            status.Text = sprintf('Selesai dalam %.2f detik.%s', toc(started), detail);
        catch err
            status.Text = 'Proses gagal.';
            uialert(fig, err.message, 'Gagal memproses citra');
        end
    end

    function detail = showGroundTruth(labels)
        [imageFolder, name] = fileparts(imagePath);
        truthPath = fullfile(imageFolder, '..', 'labels', [name '.png']);
        if ~isfile(truthPath)
            cla(semanticAxes(4));
            title(semanticAxes(4), 'Ground truth tidak tersedia');
            detail = '';
            return
        end
        truth = imread(truthPath);
        showImage(semanticAxes(4), colorizeLabels(truth, classColors), 'Ground truth');
        labeled = truth ~= 255;
        detail = sprintf(' Akurasi piksel terhadap ground truth: %.1f%%.', ...
            100 * mean(labels(labeled) == truth(labeled)));
    end
end

function rgb = colorizeLabels(labels, classColors)
palette = [classColors; 0 0 0];
index = double(labels) + 1;
index(labels == 255) = size(palette, 1);
rgb = uint8(reshape(palette(index, :), [size(labels) 3]));
end

function color = readableTextColor(background)
if [0.299 0.587 0.114] * background(:) > 140
    color = [0 0 0];
else
    color = [1 1 1];
end
end

function [controls, axesList, tab] = makeTab(tabs, name, titles, columns, callback)
tab = uitab(tabs, 'Title', name);
layout = uigridlayout(tab, [3 1]);
layout.RowHeight = {70, 30, '1x'};
controls = uigridlayout(layout, [2 columns]);
controls.RowHeight = {22, 24};
controls.Padding = [0 0 0 0];
uibutton(layout, 'Text', 'Proses', 'ButtonPushedFcn', callback);
images = uigridlayout(layout, [1 numel(titles)]);
axesList = gobjects(1, numel(titles));
for k = 1:numel(titles)
    axesList(k) = uiaxes(images);
    title(axesList(k), titles{k});
    axis(axesList(k), 'off');
end
end

function showImage(ax, img, name)
if islogical(img), img = uint8(img) * 255; end
if ismatrix(img), img = repmat(img, 1, 1, 3); end
image(ax, img);
axis(ax, 'image');
axis(ax, 'off');
title(ax, name, 'Interpreter', 'none');
end
