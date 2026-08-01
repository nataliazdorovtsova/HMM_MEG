%% Cut out bad segments with OSL view

addpath(genpath('/imaging/local/software/spm_cbu_svn/releases/spm12_latest/')); % SPM v12
setenv('OSLDIR','/imaging/local/software/spm_toolbox/osl/');
OSLDIR = '/imaging/local/software/spm_toolbox/osl/';
addpath('/imaging/local/software/EEGlab/');

% Load file
D = spm_eeg_load('/imaging/astle/users/nz01/RED_MEG/Scripts/SPM_Files/bffilteredspmeeg_99125_cleanica_sss');
modalities = {'MEGMAG','MEGPLANAR'};

% look for bad segments
%D = osl_detect_artefacts(D,'badchannels',false,'badtimes',true,'modalities',modalities);
view = oslview(D);
D.save;

%% EEGLAB




