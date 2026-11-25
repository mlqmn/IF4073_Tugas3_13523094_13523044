function labels = SemanticSegment(imagePath)
folder = fileparts(mfilename('fullpath'));
outputPath = [tempname '.png'];
command = sprintf('"%s" "%s" "%s" "%s"', pythonExecutable(folder), ...
    fullfile(folder, 'semantic', 'predict.py'), imagePath, outputPath);
if isunix
    % MATLAB prepends its bundled libstdc++ to LD_LIBRARY_PATH, which makes importing PyTorch fail.
    command = ['env -u LD_LIBRARY_PATH ' command];
else
    % cmd.exe /c strips the first and last quote of the line, so the quoted paths need an outer pair.
    command = ['"' command '"'];
end
[exitCode, output] = system(command);
if exitCode ~= 0
    error('Segmentasi semantik gagal:\n%s', strtrim(output));
end
labels = imread(outputPath);
delete(outputPath);
end

function python = pythonExecutable(folder)
candidates = {fullfile(folder, '.venv', 'bin', 'python'), ...
    fullfile(folder, '.venv', 'Scripts', 'python.exe')};
python = 'python';
for k = 1:numel(candidates)
    if isfile(candidates{k})
        python = candidates{k};
        return
    end
end
end
