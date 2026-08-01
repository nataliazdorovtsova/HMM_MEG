%% parcellate into 38-parcel representation
% 
% This is a complete representation of all possible parcellations. Note that parcels may be
% * Weighted - A voxel may be assigned to a parcel with a weighting factor
% * Overlapping- A voxel may belong to multiple parcels

% To load a parcellation, create a parcellation object providing input as a '.nii' file. The
% .nii file should contain a matrix in one of the 4 supported sizes.
setenv('OSLDIR','/imaging/local/software/spm_toolbox/osl/');
OSLDIR = '/imaging/local/software/spm_toolbox/osl/parcellations';
p = parcellation(fullfile(OSLDIR,'fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz'));

%%
% * weight_mask: The XYZ x parcels representation of the parcellation
% * template_mask: The background structural image/mask
% * template_coordinates: The MNI coordinates for each voxel 
% * template_fname: The filename of the standard mask 
% * labels: If provided, the names of each ROI
% * is_weighted: true if the voxels are weighted
% * is_overlapping: true if any voxel belongs to more than one parcel
% * resolution: spatial resolution of the standard mask
% * n_parcels: number of parcels in the parcellation
% * n_voxels: number of voxels in the mask

%% Reshaping matrices
%
% One the most basic operations is converting between the XYZ x parcels and Voxels x parcels
% representations. You can do this with the |to_matrix()| and |to_vol()| methods. For example
matrix_representation = p.to_matrix(p.weight_mask);
size(matrix_representation)
volume_representation = p.to_vol(matrix_representation);
size(volume_representation)

%%
% These functions also support an additional usage pattern. Often it is useful to visualize
% parcel-based data on the brain - for example, if you know the activation or spectral power
% at the parcel level. The data then consists of a vector, parcels x 1 (or 1 x parcels). Both
% |to_matrix()| and |to_vol()| can be given such a vector, which will then be expanded onto the 
% voxels in either the matrix or volume representation. 
expanded_volume = p.to_matrix(1:38);
size(expanded_volume)

%%
% Note that this can only be performed if the parcellation is binary (unweighted, with no overlap).
% If the parcellation does not meet these requirements, it will automatically be converted, and a 
% warning will be displayed to indicate that this has occurred. 
%
% Finally, it also possible to convert the parcellation to the 'Voxels x 1' representation where
% value indicates parcel assignment
v = p.value_vector;

%% MNI coordinates
% The MNI coordinates for the template are stored in the |template_coordinates| property.
size(p.template_coordinates)

%%
% See for example
figure
scatter3(p.template_coordinates(:,1),p.template_coordinates(:,2),p.template_coordinates(:,3))
axis equal
set(gca,'View', [-117.5000   26.8000])
xlabel('X');
ylabel('Y');
zlabel('Z');

%%
% The coordinates for each parcel can be obtained using the |roi_coordinates| method
r = p.roi_coordinates;

%%
% which returns a cell array of matrices, where each matrix contains the MNI coordinates for the voxels
% belonging to the parcel. 
class(r)
size(r)
size(r{1})

%%
% Finally, you can return the centre-of-mass of each parcel (the average of the ROI coordinates for 
% voxels belonging to the parcel) using the |roi_centres| method
c = p.roi_centers;
c(1:3,:)
hold on
scatter3(c(:,1),c(:,2),c(:,3),50,'ro','filled')

%% Binarizing
% We saw above that some operations required the parcellation to be binary. You can obtain the weight
% matrix corresponding to a binary parcellation using the |binarize()| function
binary_mask = p.binarize();

%% remove overlap

no_overlap = p.remove_overlap();

%% remove weights

unweighted = p.remove_weights();

%% binarise

binary_parcellation = parcellation(p.binarize);

%% Usage with ROI-nets
% MEG-ROI-nets can compute parcel timecourses based on voxel timecourses, using methods such as PCA to 
% reduce dimensionality. This functionality is provided through |ROInets.get_node_tcs()|. The parcellation
% needs to be passed in as a binary Voxels x parcels representation. As a shortcut, this can be obtained
% using the |parcelflag()| method. For example, to compute parcel timecourses from an SPM object, use

addpath(genpath('/imaging/local/software/spm_cbu_svn/releases/spm12_latest'));
addpath(genpath('/imaging/astle/users/nz01/RED_MEG/Scripts/MEG-ROI-nets/'));

D = spm_eeg_load('/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_calm06_cleanica_sss');
has_montage(D)
D = D.montage('switch',3) % CHECK THIS!!!!!!

%% convert file to parcellation format

D = get_node_tcs(D,binary_parcellation.parcelflag,'pca')
D.save();

%% change montages

Data = spm_eeg_load('/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_calm06_cleanica_sss');
has_montage(Data) 
parcel_montage=4; % change per participant
Data = Data.montage('switch',parcel_montage);
Data.save();

%% Plotting and visualization in Matlab
% The Parcellation object provides a number of options for plotting. To start with, the parcellation can be
% plotting using
p = binary_parcellation;
p.plot

%%
% This displays a 3D plot of the parcellation. Each ROI can be selected from the dropdown list. 
% It is also possible to show spatial maps of volume-wise activation - for example,
% a power map, or the activation map for an HMM state. 
p.plot_activation(rand(size(p.template_mask)));

%%
% The input should be in XYZ x 1 format, but it will be automatically expanded if provided in Parcels x 1
% format. For example
p.plot_activation(rand(p.n_parcels,1));

%%
% Lastly, if you have a brain network connectivity matrix, you can display the strongest connections
% using the |plot_network| method. For example, to plot the top 5% of connections, you can use
connection_matrix = randn(p.n_parcels);
size(connection_matrix)
[h_patch,h_scatter] = p.plot_network(connection_matrix,0.95);

%% Plotting using osleyes
% There are a number of plotting options using osleyes. These can be accessed through the |osleyes|
% method. By default, this will display the parcellation with one volume for each parcel e.g.
p.osleyes

%%
% The |osleyes| method allows you to pass in a matrix to be displayed. For example,

p.osleyes(m)

%%
% where the |m| matrix will be expanded into volume format if required. To plot all parcels in the
% same volume, you can use
p.osleyes(p.value_vector)

% To plot the parcellation after binarization, you can use
p.osleyes(p.binarize)

%% Saving nii files
% Lastly, and perhaps most importantly, you can save a matrix to a .nii file using
p.savenii(p.weight_mask,'parcel38_parcel.nii')

%%
% this will create a file 'filename.nii.gz'. The weight mask is written
% directly into the .nii file, so it may only make sense if you pass in a
% volume. For example, to make a .nii file with the XYZ x 1 representation of
% the parcellation (value indices parcel membership) you could use
p.savenii(p.to_vol(1:38),'parcel38_volume');
