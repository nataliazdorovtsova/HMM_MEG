%% Beamforming with OSL

% type 'cbufsl' into new terminal
% also get SPM running
addpath('/imaging/astle/users/nz01/RED_MEG/Scripts/');
datadir = '/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files';
setenv('FSLDIR','/imaging/local/software/fsl/v5.0.7/x86_64/fsl');
disp( getenv('FSLDIR') )
osl_startup('/imaging/local/software/spm_toolbox/osl','shared')

%% LOAD DATA FILE

Data = spm_eeg_load(fullfile(datadir,'filteredspmeeg_calm06_cleanica_sss'));
Info = load('filteredspmeeg_calm06_cleanica_sss.mat');

%% Make structure

A = struct;
A.D = Data;

%%
% The MEEG object must be in a sensor-space montage - this could be the raw
% data (montage 0), or it could be an online montage obtained after running
% AFRICA

Data = Data.montage('switch',0);

%% Sensor-type normalisation
% If we were working with MEGIN Neuromag data then we would have two sensor
% types, planar gradiometers and magnetometers. Before beamforming, we
% would need to normalise these two sensor types so that they can
% contribute equally to the beamformer calculation. Briefly, this is done
% by scaling the different sensor types to so that their variances over
% time are equal.Parcellation is performed using the ROInets.get_node_tcs function within the ROInets module of OSL.
%
% Here, we are working with CTF data, where there is one sensor type - so
% this step is not strictly necessary:

A=[];
A.D=Data;
A.datatype='neuromag';
%A.datatype='ctf';
A.do_plots=true; 
[Data pcadim] = osl_normalise_sensor_data(A);

%% Check coregistration and forward model

forwardmodel = Data.inv{1}

%% Calling |osl_inverse_model|
%This takes three arguments: 1) an MEEG object, 2) a set of MNI
% coordinates to compute voxel timecourses at, and 3) an optional settings
% structure, used to override the default settings in |osl_inverse_model|.

spatial_res=8; % Spatial resolution of the voxel grid to beamform to, in mm
p = parcellation(spatial_res);
mni_coords = p.template_coordinates;

% Either way, mni_coords should be an |n_voxels x 3| matrix.

clear S
S = struct;
S.timespan          = [0 (Info.D.Nsamples/1000)]; % in secs
S.pca_order         = 50;
S.inverse_method    = 'beamform';
S.type              = 'Scalar'; % beamformer output will be a scalar (rather than a 3D vector)
S.prefix            = 'bf'; % add prefix
S.modalities        = {'MEGPLANAR','MEGMAG'};
S.fuse              = 'meg';
S.conditions = 'all';
S.mode = 'volumetric'; 


normalise_data_montage=1;
Data = Data.montage('switch',normalise_data_montage);

Data = osl_inverse_model(Data,mni_coords,S); % BEAMFORMING

%% Beamformer output
% Note that the |Data| object has now got a number of channels equal to
% the number of MNI coordinates we specified in mni_coords, and they will be 
% in the same order:

Data;

disp('Number of channels in Data:')
disp(Data.nchannels)

disp('Number of MNI coordinates:')
disp(size(mni_coords,1))

%%
% You'll see that the result of running the beamformer is the addition of
% two new online montages corresponding to the beamformed result: 
%
% * Source space (MEG) without weights normalisation,
% * Source space (MEG) with weights normalisation 
%
% Weights normalisation is used to correct the fact
% that, with beamforming, voxels that are deeper in the brain tend to have
% higher variance.

has_montage(Data)

%% 
% Switch to the montage that corresponds to the source recon with weights
% normalisation, check that source_recon_montage is set accordingly before
% running this next bit

source_recon_montage=3;
Data=Data.montage('switch',source_recon_montage)
