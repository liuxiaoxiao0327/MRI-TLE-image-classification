% Description: This MATLAB script visualizes statistical data on the cortical surface.

% NOTE: The file paths must be updated to your specific file locations.
% This script assumes you are using MATLAB R2019a or a compatible version.

%% Read the cortical thickness file for a template.
% The 'read_curv' function reads the surface data.
[thickness, fnum] = read_curv('location/lh.thickness'); [cite: 120, 121]

%% Read the brain parcellation annotation file.
% The 'read_annotation' function reads the labels and color table for different brain regions.
[vertices, label, colortable] = read_annotation('location/lh.aparc.annot'); [cite: 122, 123]

%% Read the statistical results from an Excel file.
% 'xlsread' reads the data, assuming it's a 34x1 vector of statistics.
stats = xlsread('location/stats.xlsx'); [cite: 125, 126]

% The first and fifth regions (unknown and corpus callosum) are typically
% not included in statistical analysis. This line inserts zeros in their place.
stats = [0; stats(1:3); 0; stats(4:34)]; [cite: 127]

%% Initialize a new array to hold the statistical values for each vertex.
lh_stats = zeros(size(thickness)); [cite: 128]

%% Loop through each brain region and assign the corresponding statistical value to its vertices.
for i = 1:length(colortable.struct_names) [cite: 129]
    % Find all vertices that belong to the current brain region (label).
    idx = label == colortable.table(i,5); [cite: 130]
    
    % Assign the statistical value for this region to the corresponding vertices.
    lh_stats(idx) = stats(i); [cite: 131]
end

%% Save the new file with the statistical values.
% The 'write_curv' function saves the data in a format viewable by Freeview.
write_curv('/home/alex/lh.stats', lh_stats, fnum); [cite: 134]

% You can now open this file in Freeview to visualize the statistical results on the cortical surface.
% Remember to add the FreeSurfer .m files to your MATLAB path if the functions are not found.